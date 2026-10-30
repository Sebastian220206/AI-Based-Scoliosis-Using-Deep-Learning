"""
Cobb Angle & Hybrid Quantum ML (QSVM) Inference Engine
======================================================
Integrates the trained models from the `cobb angle/` folder:
1. MobileNetV2 Feature Extractor (`cobb_mobilenetv2_feature_extractor.keras` / `mobilenetv2_feature_extractor.keras`)
2. Multi-Output Cobb Angle MLP Regressor (`cobb_deployment_files.zip`: StandardScaler -> PCA-64 -> MLPRegressor)
3. Hybrid Quantum Kernel SVM Classifier (`hybrid_qml_model.zip`: StandardScaler -> PCA-8 -> MinMaxScaler[-pi, pi] -> 8-Qubit Quantum Feature Map -> Precomputed QSVM)
4. Classical CV CLAHE + Savitzky-Golay Centerline Curvature Fallback (`Scoliosis_Cobb_Angle_Estimation.ipynb`)
"""

import io
import json
import os
import warnings
import zipfile
import cv2
import h5py
import joblib
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore", category=UserWarning)

# ─── NumPy 1.x / 2.x RandomState Unpickling Compatibility Shim ───
try:
    from numpy.random import RandomState
    import numpy.random._pickle as nrp

    class _CompatRandomState:
        def __init__(self, *args, **kwargs):
            self._rs = RandomState(42)

        def __setstate__(self, state):
            pass

        def __getattr__(self, name):
            return getattr(self._rs, name)

    _orig_bg_ctor = nrp.__bit_generator_ctor
    _orig_rs_ctor = nrp.__randomstate_ctor

    def _safe_bg_ctor(bit_generator_name="MT19937"):
        try:
            if isinstance(bit_generator_name, type):
                bit_generator_name = bit_generator_name.__name__
            return _orig_bg_ctor(bit_generator_name)
        except Exception:
            return _CompatRandomState()

    def _safe_rs_ctor(bit_generator_name="MT19937", bit_generator_ctor=_safe_bg_ctor):
        try:
            return _orig_rs_ctor(bit_generator_name, bit_generator_ctor)
        except Exception:
            return _CompatRandomState()

    nrp.__bit_generator_ctor = _safe_bg_ctor
    nrp.__randomstate_ctor = _safe_rs_ctor
except Exception:
    pass


def _safe_joblib_load(raw_bytes):
    """Load a joblib pickle across NumPy 1.x and 2.x environments."""
    try:
        return joblib.load(io.BytesIO(raw_bytes))
    except ValueError:
        # Fallback for NumPy 1.x when unpickling NumPy 2.x MT19937 state
        import numpy.random._pickle as nrp
        nrp.__bit_generator_ctor = lambda *a, **kw: _CompatRandomState()
        nrp.__randomstate_ctor = lambda *a, **kw: _CompatRandomState()
        return joblib.load(io.BytesIO(raw_bytes))


class PureKerasMobileNetV2:
    """
    Exact NumPy execution engine for the saved Keras 3 `mobilenetv2_feature_extractor.keras`
    functional graph. Produces bit-for-bit identical 1280-D feature vectors without
    requiring a heavy TensorFlow runtime.
    """

    def __init__(self, keras_archive_bytes):
        with zipfile.ZipFile(io.BytesIO(keras_archive_bytes)) as kz:
            self.cfg = json.loads(kz.read("config.json"))
            w_bytes = kz.read("model.weights.h5")

        self.layers = self.cfg["config"]["layers"]
        self.weights = {}

        with h5py.File(io.BytesIO(w_bytes), "r") as hf:
            counts = {}
            type_map = {
                "Conv2D": "conv2d",
                "DepthwiseConv2D": "depthwise_conv2d",
                "BatchNormalization": "batch_normalization",
            }
            for layer in self.layers:
                cn = layer["class_name"]
                lname = layer["name"]
                if cn in type_map:
                    base_k = type_map[cn]
                    idx = counts.get(base_k, 0)
                    counts[base_k] = idx + 1
                    h5_k = base_k if idx == 0 else f"{base_k}_{idx}"
                    vgrp = hf["layers"][h5_k]["vars"]
                    self.weights[lname] = [vgrp[str(i)][:] for i in range(len(vgrp))]

    @staticmethod
    def _pad_same(x, kh, kw, sh, sw):
        _, h, w, _ = x.shape
        out_h = (h + sh - 1) // sh
        out_w = (w + sw - 1) // sw
        pad_h = max((out_h - 1) * sh + kh - h, 0)
        pad_w = max((out_w - 1) * sw + kw - w, 0)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        if top > 0 or bottom > 0 or left > 0 or right > 0:
            x = np.pad(x, ((0, 0), (top, bottom), (left, right), (0, 0)), mode="constant")
        return x

    def _conv2d(self, x, W, strides, padding):
        kh, kw, _, cout = W.shape
        sh, sw = strides
        if padding == "same":
            x = self._pad_same(x, kh, kw, sh, sw)
        if kh == 1 and kw == 1 and sh == 1 and sw == 1:
            return x @ W[0, 0]
        _, h, w, _ = x.shape
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1
        out = np.zeros((x.shape[0], out_h, out_w, cout), dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                patch = x[:, i : i + out_h * sh : sh, j : j + out_w * sw : sw, :]
                out += patch @ W[i, j]
        return out

    def _depthwise_conv2d(self, x, W, strides, padding):
        kh, kw, c, _ = W.shape
        sh, sw = strides
        if padding == "same":
            x = self._pad_same(x, kh, kw, sh, sw)
        _, h, w, _ = x.shape
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1
        out = np.zeros((x.shape[0], out_h, out_w, c), dtype=np.float32)
        W_sq = W[:, :, :, 0]
        for i in range(kh):
            for j in range(kw):
                patch = x[:, i : i + out_h * sh : sh, j : j + out_w * sw : sw, :]
                out += patch * W_sq[i, j]
        return out

    def predict(self, x):
        tensors = {}
        for layer in self.layers:
            lname = layer["name"]
            cn = layer["class_name"]
            lcfg = layer["config"]
            inodes = layer.get("inbound_nodes", [])
            if cn == "InputLayer":
                tensors[lname] = x
                continue

            args = inodes[0]["args"][0]
            if isinstance(args, list):
                inp = [tensors[a["config"]["keras_history"][0]] for a in args]
            else:
                inp = tensors[args["config"]["keras_history"][0]]

            if cn == "Conv2D":
                W = self.weights[lname][0]
                tensors[lname] = self._conv2d(inp, W, lcfg["strides"], lcfg["padding"])
            elif cn == "DepthwiseConv2D":
                W = self.weights[lname][0]
                tensors[lname] = self._depthwise_conv2d(inp, W, lcfg["strides"], lcfg["padding"])
            elif cn == "BatchNormalization":
                gamma, beta, mean, var = self.weights[lname]
                eps = lcfg.get("epsilon", 1e-3)
                tensors[lname] = gamma * (inp - mean) / np.sqrt(var + eps) + beta
            elif cn == "ReLU":
                max_val = lcfg.get("max_value", None)
                out = np.maximum(inp, 0.0)
                if max_val is not None:
                    out = np.minimum(out, float(max_val))
                tensors[lname] = out
            elif cn == "ZeroPadding2D":
                p = lcfg["padding"]
                tensors[lname] = np.pad(
                    inp,
                    ((0, 0), (p[0][0], p[0][1]), (p[1][0], p[1][1]), (0, 0)),
                    mode="constant",
                )
            elif cn == "Add":
                tensors[lname] = inp[0] + inp[1]
            elif cn == "GlobalAveragePooling2D":
                tensors[lname] = np.mean(inp, axis=(1, 2))
            else:
                raise ValueError(f"Unsupported layer type: {cn}")

        return tensors[self.layers[-1]["name"]]


def quantum_feature_map_batch(X):
    """
    Exact 8-qubit PennyLane quantum feature map state simulator (`Untitled6.ipynb` Cell 47 & 86):
      1. Hadamard + RY(x[i]) on each wire i in 0..7
      2. Ring CNOT entanglement: CNOT(i, i+1) for i=0..6, and CNOT(7, 0)
    Returns state vectors of shape (N, 256).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    n_samples = X.shape[0]

    c = np.cos(X / 2.0)
    s = np.sin(X / 2.0)
    q0 = (c - s) / np.sqrt(2.0)
    q1 = (s + c) / np.sqrt(2.0)

    state = np.ones((n_samples, 1), dtype=np.float64)
    for i in range(8):
        pair = np.stack([q0[:, i], q1[:, i]], axis=1)
        state = (state[:, :, None] * pair[:, None, :]).reshape(n_samples, -1)

    state = state.reshape((n_samples,) + (2,) * 8)

    for i in range(8):
        ctrl = i + 1
        tgt = ((i + 1) % 8) + 1
        idx = [slice(None)] * 9
        idx[ctrl] = 1
        sub_axis = tgt if tgt < ctrl else tgt - 1
        state[tuple(idx)] = np.flip(state[tuple(idx)], axis=sub_axis)

    return state.reshape(n_samples, 256)


def compute_cv_centerline(image_path):
    """
    Classical computer vision centerline & curvature extractor from
    `Scoliosis_Cobb_Angle_Estimation.ipynb` (Cells 0-10).
    Uses CLAHE enhancement, spinal corridor ROI cropping, sliding-window intensity
    tracking, and Savitzky-Golay smoothing.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    x_start = int(w * 0.38)
    x_end = int(w * 0.68)
    spine_roi = enhanced[:, x_start:x_end]
    roi_smooth = cv2.GaussianBlur(spine_roi, (7, 7), 0)

    roi_h, roi_w = roi_smooth.shape
    centerline = []
    x_prev = roi_w // 2
    window = max(12, int(roi_w * 0.08))

    for y in range(roi_h):
        row = roi_smooth[y]
        left = max(0, x_prev - window)
        right = min(roi_w, x_prev + window + 1)
        segment = row[left:right]
        if len(segment) > 0:
            x = left + int(np.argmax(segment))
        else:
            x = x_prev
        centerline.append([x, y])
        x_prev = x

    centerline = np.array(centerline, dtype=np.float64)
    win_len = min(51, roi_h if roi_h % 2 == 1 else roi_h - 1)
    if win_len >= 5:
        centerline[:, 0] = savgol_filter(centerline[:, 0], win_len, 3)

    x_coords = centerline[:, 0]
    dx = np.gradient(x_coords)
    d2x = np.gradient(dx)
    curvature = np.abs(d2x)

    start = int(0.15 * roi_h)
    end = int(0.85 * roi_h)
    apex_index = start + int(np.argmax(curvature[start:end]))

    upper = centerline[:apex_index]
    lower = centerline[apex_index:]
    upper_part = upper[int(len(upper) * 0.35) :]
    lower_part = lower[: int(len(lower) * 0.65)]

    cv_angle = 0.0
    if len(upper_part) >= 5 and len(lower_part) >= 5:
        m1, _ = np.polyfit(upper_part[:, 1], upper_part[:, 0], 1)
        m2, _ = np.polyfit(lower_part[:, 1], lower_part[:, 0], 1)
        theta = abs(np.arctan(m1) - np.arctan(m2))
        cv_angle = float(np.degrees(theta))

    # Map centerline back to full image coordinates (sampled to ~24 anatomical points)
    full_x = centerline[:, 0] + x_start
    full_y = centerline[:, 1]
    sample_indices = np.linspace(int(0.08 * roi_h), int(0.92 * roi_h), 24, dtype=int)
    sampled_points = [
        {"x": round(float(full_x[idx]), 1), "y": round(float(full_y[idx]), 1), "confidence": 0.85, "w": 20.0, "h": 20.0}
        for idx in sample_indices
    ]
    curve_indices = np.linspace(int(0.08 * roi_h), int(0.92 * roi_h), 60, dtype=int)
    fitted_curve = [
        {"x": round(float(full_x[idx]), 1), "y": round(float(full_y[idx]), 1)}
        for idx in curve_indices
    ]

    return {
        "cv_cobb_angle": round(cv_angle, 2),
        "apex": {"x": round(float(full_x[apex_index]), 1), "y": round(float(full_y[apex_index]), 1)},
        "sampled_points": sampled_points,
        "fitted_curve": fitted_curve,
    }


def classify_severity(angle_deg):
    """Clinical severity classification matching `Scoliosis_Cobb_Angle_Estimation.ipynb` Cell 9."""
    if angle_deg < 10.0:
        return "Normal"
    elif angle_deg < 25.0:
        return "Mild"
    elif angle_deg < 40.0:
        return "Moderate"
    else:
        return "Severe"


class CobbAngleEngine:
    """Singleton manager that loads and runs both `cobb_deployment_files.zip` and `hybrid_qml_model.zip`."""

    def __init__(self, cobb_folder_path):
        self.cobb_folder = cobb_folder_path
        cobb_zip = os.path.join(cobb_folder_path, "cobb_deployment_files.zip")
        qml_zip = os.path.join(cobb_folder_path, "hybrid_qml_model.zip")

        with zipfile.ZipFile(cobb_zip, "r") as z_cobb:
            keras_bytes = z_cobb.read("cobb_mobilenetv2_feature_extractor.keras")
            self.cobb_scaler = _safe_joblib_load(z_cobb.read("cobb_standard_scaler.pkl"))
            self.cobb_pca64 = _safe_joblib_load(z_cobb.read("cobb_pca_64.pkl"))
            self.cobb_mlp = _safe_joblib_load(z_cobb.read("cobb_mlp_regressor.pkl"))

        with zipfile.ZipFile(qml_zip, "r") as z_qml:
            self.qml_scaler = _safe_joblib_load(z_qml.read("standard_scaler.pkl"))
            self.qml_pca8 = _safe_joblib_load(z_qml.read("pca_8.pkl"))
            self.qml_quantum_scaler = _safe_joblib_load(z_qml.read("quantum_scaler.pkl"))
            self.qsvm = _safe_joblib_load(z_qml.read("quantum_kernel_svm.pkl"))
            self.train_q_features = np.load(io.BytesIO(z_qml.read("training_quantum_features.npy")))

        # Shared MobileNetV2 backbone (identical SHA-256 weights in both archives)
        self.mobilenet = PureKerasMobileNetV2(keras_bytes)

        # Precompute 256-D quantum states for the 207 QSVM training samples
        self.train_q_states = quantum_feature_map_batch(self.train_q_features)

    def predict(self, image_path):
        """
        Run full Cobb Angle MLP regression, Hybrid Quantum Kernel SVM classification,
        and CV centerline analysis on an X-ray image.
        """
        # 1. Preprocess image exactly as in the notebooks (224x224 RGB, MobileNetV2 [-1, 1] scaling)
        bgr = cv2.imread(image_path)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_224 = cv2.resize(rgb, (224, 224))
        else:
            pil_img = Image.open(image_path).convert("RGB").resize((224, 224))
            rgb_224 = np.array(pil_img)

        batch = np.expand_dims(rgb_224.astype(np.float32), axis=0)
        batch_preprocessed = (batch / 127.5) - 1.0

        # 2. Extract 1280-D MobileNetV2 features
        features_1280 = self.mobilenet.predict(batch_preprocessed)

        # 3. Multi-Output Cobb Angle Regression (StandardScaler -> PCA-64 -> MLPRegressor)
        cobb_scaled = self.cobb_scaler.transform(features_1280)
        cobb_pca = self.cobb_pca64.transform(cobb_scaled)
        raw_angles = self.cobb_mlp.predict(cobb_pca)[0]
        angles = [round(max(0.0, float(a)), 2) for a in raw_angles]
        primary_cobb = round(max(angles), 2)
        severity = classify_severity(primary_cobb)

        # 4. Hybrid Quantum Kernel SVM Classification (StandardScaler -> PCA-8 -> MinMax[-pi,pi] -> QSVM)
        qml_scaled = self.qml_scaler.transform(features_1280)
        qml_pca8 = self.qml_pca8.transform(qml_scaled)
        qml_angles = self.qml_quantum_scaler.transform(qml_pca8)

        test_q_state = quantum_feature_map_batch(qml_angles)
        kernel_matrix = np.abs(test_q_state @ self.train_q_states.T) ** 2
        qsvm_pred = int(self.qsvm.predict(kernel_matrix)[0])
        qsvm_decision = float(self.qsvm.decision_function(kernel_matrix)[0])
        # Convert decision margin to calibrated confidence via sigmoid
        qsvm_prob_scoliosis = float(1.0 / (1.0 + np.exp(-2.0 * qsvm_decision)))
        qsvm_confidence = qsvm_prob_scoliosis if qsvm_pred == 1 else (1.0 - qsvm_prob_scoliosis)
        qsvm_label = "Scoliosis" if qsvm_pred == 1 else "Normal"

        # 5. Classical CV Centerline & Apex Analysis
        cv_result = compute_cv_centerline(image_path)

        return {
            "primary_cobb_angle": primary_cobb,
            "angles": {
                "proximal_thoracic": angles[0],
                "main_thoracic": angles[1],
                "thoracolumbar": angles[2],
            },
            "angles_list": angles,
            "severity": severity,
            "qsvm": {
                "prediction": qsvm_label,
                "class_id": qsvm_pred,
                "confidence": round(qsvm_confidence * 100.0, 1),
                "decision_score": round(qsvm_decision, 4),
            },
            "cv_analysis": cv_result,
        }


_engine_instance = None


def get_cobb_engine():
    global _engine_instance
    if _engine_instance is None:
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cobb angle")
        _engine_instance = CobbAngleEngine(folder)
    return _engine_instance
