from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from inference_sdk import InferenceHTTPClient
import os
import numpy as np
from PIL import Image

UPLOAD_FOLDER = "uploads"
GLB_FOLDER = os.path.join(os.path.dirname(__file__), "spine glb")
# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
CORS(app)   # Enable CORS

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="wC8micD3mh1ekXVn0kTz"
)

@app.route("/")
def home():
    return "Scoliosis AI Backend Running"

@app.route("/models/spine.stl")
def serve_spine_model():
    return send_from_directory(GLB_FOLDER, "Spine.stl")

@app.route("/models/spine.gltf")
def serve_spine_gltf():
    return send_from_directory(GLB_FOLDER, "_Spine.gltf")

@app.route("/models/spine.glb")
def serve_spine_glb():
    return send_from_directory(GLB_FOLDER, "Spine_NIH3D.glb")

@app.route("/models/vertebrae/<path:filename>")
def serve_vertebra_model(filename):
    full_path = os.path.join(GLB_FOLDER, filename)
    if os.path.exists(full_path):
        return send_from_directory(GLB_FOLDER, filename)
    encoded_filename = filename.replace(" ", "%20")
    if os.path.exists(os.path.join(GLB_FOLDER, encoded_filename)):
        return send_from_directory(GLB_FOLDER, encoded_filename)
    return send_from_directory(GLB_FOLDER, filename)

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["image"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Get image dimensions for proper scaling
    img = Image.open(filepath)
    img_width, img_height = img.size

    result = client.run_workflow(
        workspace_name="acoustics-workspace",
        workflow_id="custom-workflow-2",
        images={"image": filepath},
        use_cache=True
    )
    
    print(f"Roboflow raw response: {result}")
    
    # Robust detection extraction
    try:
        # Check if result is a list (standard workflow output)
        if isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, dict):
                predictions_obj = item.get("predictions", {})
                if isinstance(predictions_obj, list):
                    detections = predictions_obj
                else:
                    detections = predictions_obj.get("predictions", [])
            else:
                detections = []
        elif isinstance(result, dict):
            # Try common Roboflow keys
            detections = result.get("predictions", [])
            if isinstance(detections, dict):
                detections = detections.get("predictions", [])
        else:
            detections = []
    except Exception as e:
        print(f"Error extracting detections: {e}")
        detections = []

    total_raw = len(detections)
    print(f"Detections found: {total_raw}")

    # 1. Filter by confidence
    MIN_CONFIDENCE = 0.20
    detections = [d for d in detections if d.get("confidence", 0) >= MIN_CONFIDENCE]

    # 2. Filter out tiny detections (noise)
    detections = [d for d in detections if d.get("width", 0) > 5 and d.get("height", 0) > 5]

    # 3. Sort top to bottom by Y
    detections = sorted(detections, key=lambda d: d["y"])

    # 4. Remove duplicate detections (merge close points, keep higher confidence)
    MERGE_DISTANCE = 15
    filtered = []
    for d in detections:
        is_duplicate = False
        for existing in filtered:
            if abs(d["x"] - existing["x"]) < MERGE_DISTANCE and abs(d["y"] - existing["y"]) < MERGE_DISTANCE:
                if d.get("confidence", 0) > existing.get("confidence", 0):
                    filtered.remove(existing)
                    filtered.append(d)
                is_duplicate = True
                break
        if not is_duplicate:
            filtered.append(d)

    detections = sorted(filtered, key=lambda d: d["y"])

    # 5. IQR-based outlier removal on X-axis
    if len(detections) >= 5:
        x_vals = [d["x"] for d in detections]
        q1 = np.percentile(x_vals, 25)
        q3 = np.percentile(x_vals, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        detections = [d for d in detections if lower <= d["x"] <= upper]

    # Raw detected vertebra points
    raw_points = [
        {
            "x": round(d["x"], 1),
            "y": round(d["y"], 1),
            "confidence": round(d.get("confidence", 0), 3),
            "w": round(d.get("width", 0), 1),
            "h": round(d.get("height", 0), 1)
        }
        for d in detections
    ]

    # 6. Polynomial spine fitting (fit X as function of Y for vertical spine)
    fitted_points = []
    if len(detections) >= 3:
        y_arr = np.array([d["y"] for d in detections])
        x_arr = np.array([d["x"] for d in detections])

        # Fit polynomial: degree based on number of points
        degree = min(4, len(detections) - 1)
        coeffs = np.polyfit(y_arr, x_arr, degree)
        poly = np.poly1d(coeffs)

        # Generate evenly spaced points along the spine
        num_interpolated = max(20, len(detections) * 3)
        y_smooth = np.linspace(y_arr.min(), y_arr.max(), num_interpolated)
        x_smooth = poly(y_smooth)

        fitted_points = [
            {"x": round(float(x_smooth[i]), 1), "y": round(float(y_smooth[i]), 1)}
            for i in range(len(y_smooth))
        ]

    return jsonify({
        "spine_points": raw_points,
        "fitted_curve": fitted_points,
        "image_size": {"width": img_width, "height": img_height},
        "total_raw": total_raw,
        "total_filtered": len(raw_points)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)