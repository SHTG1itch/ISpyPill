import base64
import io
import os
import traceback

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps

from pill_counter import analyze_reference, count_pills

# React build output (produced by: cd frontend && npm run build)
DIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")

app = Flask(__name__, static_folder=DIST_DIR, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Longest side of any uploaded image is capped at this for speed
MAX_IMAGE_DIM = 2000


# ── Image helpers ──────────────────────────────────────────────────────────────

def load_image_from_file(file_storage):
    """
    Convert a Flask FileStorage object to a numpy BGR array.
    - Corrects EXIF rotation (handles portrait/landscape mobile photos)
    - Resizes to MAX_IMAGE_DIM on the longest side for performance
    """
    in_memory = io.BytesIO(file_storage.read())
    pil_image = Image.open(in_memory)
    pil_image = ImageOps.exif_transpose(pil_image)   # fix EXIF orientation
    pil_image = pil_image.convert("RGB")

    w, h = pil_image.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale    = MAX_IMAGE_DIM / max(w, h)
        pil_image = pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def encode_image_to_base64(image_np):
    """Encode a numpy BGR image to a base64 JPEG string."""
    success, buffer = cv2.imencode(".jpg", image_np, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise RuntimeError("Failed to encode annotated image.")
    return base64.b64encode(buffer).decode("utf-8")


# ── API ────────────────────────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    if "reference_pill" not in request.files:
        return jsonify({"error": "Missing reference pill image."}), 400
    if "group_photo" not in request.files:
        return jsonify({"error": "Missing group photo image."}), 400

    ref_file   = request.files["reference_pill"]
    group_file = request.files["group_photo"]

    if not ref_file.filename:
        return jsonify({"error": "No reference pill file selected."}), 400
    if not group_file.filename:
        return jsonify({"error": "No group photo file selected."}), 400

    try:
        ref_image   = load_image_from_file(ref_file)
        group_image = load_image_from_file(group_file)

        ref_area, color_profiles, is_white, ref_shape = analyze_reference(ref_image)

        count, annotated_image = count_pills(
            group_image, ref_area, color_profiles,
            is_white=is_white, ref_shape=ref_shape,
        )

        return jsonify({
            "count":              count,
            "annotated_image":    encode_image_to_base64(annotated_image),
            "ref_area_px":        round(ref_area, 1),
            "is_white_pill":      is_white,
            "num_color_clusters": len(color_profiles),
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    except Exception:
        app.logger.error("Unexpected error:\n" + traceback.format_exc())
        return jsonify({"error": "Image processing failed. Please try again with clearer photos."}), 500


# ── Serve React SPA ────────────────────────────────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    # Serve exact static files (JS/CSS/images produced by Vite build)
    target = os.path.join(DIST_DIR, path)
    if path and os.path.isfile(target):
        return send_from_directory(DIST_DIR, path)
    # Fall back to index.html for SPA routing
    return send_from_directory(DIST_DIR, "index.html")


if __name__ == "__main__":
    if not os.path.isdir(DIST_DIR):
        print("\n⚠  React build not found.")
        print("   Run:  cd frontend && npm install && npm run build\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
