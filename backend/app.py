from flask import Flask, jsonify
from flask_cors import CORS
from google.oauth2 import service_account
from googleapiclient.discovery import build

app = Flask(__name__)
CORS(app)

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'credentials.json'
FOLDER_ID = '1g_EhUHCf1Jlqm8cTJNkavG8g0ONkE9gt'

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

service = build('drive', 'v3', credentials=credentials)


@app.route("/api/xrays")
def list_xrays():
    results = service.files().list(
        q=f"'{FOLDER_ID}' in parents and mimeType contains 'image/'",
        fields="files(id, name)"
    ).execute()

    files = results.get("files", [])

    xrays = []
    for file in files:
        xrays.append({
            "name": file["name"],
            "url": f"https://drive.google.com/thumbnail?id={file['id']}&sz=w1000"

        })

    return jsonify(xrays)


if __name__ == '__main__':
    app.run(debug=True)
