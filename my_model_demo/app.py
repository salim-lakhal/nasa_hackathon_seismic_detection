import os
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "seismic_cnn.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def load_model():
    """Load the trained CNN model at startup."""
    global model

    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model file not found at {MODEL_PATH}, using fallback detection")
        return None

    try:
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, 1))

        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        resnet.load_state_dict(state_dict)
        resnet.to(device)
        resnet.eval()
        model = resnet
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_seismic(image_path):
    """
    Run seismic detection on a spectrogram image.
    Uses the trained CNN if available, falls back to image analysis otherwise.
    Returns (label, confidence).
    """
    if model is not None:
        try:
            img = Image.open(image_path).convert("L")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                prob = torch.sigmoid(output).item()

            if prob > 0.5:
                return "Seismic Event Detected", prob
            else:
                return "No Seismic Event Detected", 1.0 - prob
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            return fallback_detection(image_path)
    else:
        return fallback_detection(image_path)


def fallback_detection(image_path):
    """Fallback detection using spectral energy analysis."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Error: could not load image", 0.0

    img_resized = cv2.resize(img, (224, 224))
    mean_intensity = np.mean(img_resized)
    std_intensity = np.std(img_resized)

    # spectrograms with seismic events have higher variance
    energy_ratio = std_intensity / (mean_intensity + 1e-6)

    if energy_ratio > 0.4:
        confidence = min(0.5 + energy_ratio * 0.3, 0.95)
        return "Seismic Event Detected", confidence
    else:
        confidence = min(0.5 + (0.4 - energy_ratio) * 0.5, 0.95)
        return "No Seismic Event Detected", confidence


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded. Please select an image.")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    if not allowed_file(file.filename):
        return render_template("index.html", error="Invalid file type. Use PNG or JPEG.")

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        label, confidence = detect_seismic(file_path)
        return render_template(
            "result.html",
            label=label,
            confidence=f"{confidence:.1%}",
            image_path=filename,
            is_seismic="Detected" in label and "No" not in label,
        )
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return render_template("index.html", error="Error processing image. Please try again.")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    load_model()
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, host="127.0.0.1", port=5000)
