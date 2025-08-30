import os
import uuid
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Import agents
import doctor as doctor_agent
import prompting as prompting_agent
import blender_agent

# ------------------------
# Flask Setup
# ------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "static/models"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "tif", "tiff"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MODEL_FOLDER"] = MODEL_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


# ------------------------
# Helpers
# ------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------------
# Routes
# ------------------------
@app.route("/")
def index():
    """Frontend"""
    return render_template("index.html")


@app.route("/health")
def health():
    """Health check"""
    return jsonify({"status": "ok"})


@app.route("/supported-formats")
def supported_formats():
    """Supported formats list"""
    return jsonify({"formats": list(ALLOWED_EXTENSIONS)})


@app.route("/upload", methods=["POST"])
def upload_scan():
    """Main upload + analysis pipeline"""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Unsupported file format"}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(upload_path)

    try:
        # ------------------
        # Step 1: Doctor Agent → analyze scan
        # ------------------
        report = doctor_agent.analyze_scan(upload_path)

        # ------------------
        # Step 2: Prompt Agent → generate Blender prompt
        # ------------------
        blender_prompt = prompting_agent.generate_prompt_with_api(report)

        # ------------------
        # Step 3: Blender Agent → generate 3D model
        # ------------------
        model_file = blender_agent.create_model(blender_prompt)

        # Ensure correct web path for frontend
        if model_file.startswith("static/"):
            model_path = "/" + model_file
        else:
            model_path = "/static/models/" + os.path.basename(model_file)

        # ------------------
        # Final Response
        # ------------------
        return jsonify({
            "success": True,
            "uploaded_file": unique_name,
            "medical_report": report,
            "blender_prompt": blender_prompt,
            "model_path": model_path
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ------------------------
# Run Server
# ------------------------
if __name__ == "__main__":
    print("Starting Medical Scan Analysis API...")
    print("=" * 50)
    print("Available endpoints:")
    print("  GET  /                  - Main application page")
    print("  POST /upload            - Upload and analyze medical scan")
    print("  GET  /health            - Health check")
    print("  GET  /supported-formats - Get supported file formats")
    print("  GET  /static/models/<file> - Serve 3D model files")
    print("=" * 50)

    app.run(host="0.0.0.0", port=5000, debug=True)
