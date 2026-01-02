from flask import Flask, send_from_directory, jsonify, request
import subprocess
import os

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = "/root/Pictures"
MODEL_COMMANDS = {
    "yolo": [
        "python",
        "/root/YOLOV5/main/yolov5/detect.py",
        "--weights",
        "/root/YOLOV5/main/yolov5/runs/train/exp21/weights/best.pt",
        "--source",
        f"{UPLOAD_FOLDER}/input.jpg",
        "--project",
        UPLOAD_FOLDER,
        "--exist-ok",
    ],
    "detr": [
        "python",
        "/root/DETR-NB/inference.py",
    ],
}
MODEL_OUTPUTS = {
    "yolo": f"{UPLOAD_FOLDER}/exp/input.jpg",
    "detr": f"{UPLOAD_FOLDER}/DETR/result.png",
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:filepath>")
def serve_static(filepath):
    return send_from_directory(BASE_DIR, filepath)


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400
    file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    return jsonify({"status": "success"})


@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json() or {}
    model = data.get("model", "yolo").lower()
    
    if model not in MODEL_COMMANDS:
        return jsonify({"error": f"Unknown model: {model}"}), 400
    
    try:
        cmd = MODEL_COMMANDS[model]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output_file = MODEL_OUTPUTS.get(model)
        return jsonify({"status": "success", "output": output_file})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Inference timeout"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/results/<path:filepath>")
def get_result(filepath):
    allowed_files = {"input.jpg", "result.png"}
    if filepath not in allowed_files:
        return jsonify({"error": "Invalid file"}), 403
    try:
        if filepath == "input.jpg":
            return send_from_directory(f"{UPLOAD_FOLDER}/exp", filepath)
        elif filepath == "result.png":
            return send_from_directory(f"{UPLOAD_FOLDER}/DETR", filepath)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
