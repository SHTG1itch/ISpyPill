import base64
import io
import traceback

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

from pill_counter import analyze_reference, count_pills

# Longest side of any uploaded image is capped at this for speed
MAX_IMAGE_DIM = 2000

app = Flask(__name__)

# Maximum upload size: 16 MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def load_image_from_file(file_storage):
    """
    Convert a Flask FileStorage object to a numpy BGR array.
    - Corrects EXIF rotation (handles portrait/landscape mobile photos)
    - Resizes to MAX_IMAGE_DIM on the longest side for performance
    """
    in_memory = io.BytesIO(file_storage.read())
    pil_image = Image.open(in_memory)
    # Fix EXIF orientation before anything else
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = pil_image.convert("RGB")

    # Resize if too large
    w, h = pil_image.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)

    np_image = np.array(pil_image)
    bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    return bgr_image


def encode_image_to_base64(image_np):
    """Encode a numpy BGR image to a base64 JPEG string."""
    success, buffer = cv2.imencode(".jpg", image_np, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise RuntimeError("Failed to encode annotated image.")
    return base64.b64encode(buffer).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # --- Validate uploaded files ---
    if "reference_pill" not in request.files:
        return jsonify({"error": "Missing reference pill image."}), 400
    if "group_photo" not in request.files:
        return jsonify({"error": "Missing group photo image."}), 400

    ref_file = request.files["reference_pill"]
    group_file = request.files["group_photo"]

    if ref_file.filename == "":
        return jsonify({"error": "No reference pill file selected."}), 400
    if group_file.filename == "":
        return jsonify({"error": "No group photo file selected."}), 400

    try:
        # --- Load images ---
        ref_image = load_image_from_file(ref_file)
        group_image = load_image_from_file(group_file)

        # --- Analyze reference pill ---
        ref_area, color_profiles, is_white, ref_shape = analyze_reference(ref_image)

        # --- Count pills in group photo ---
        count, annotated_image = count_pills(
            group_image,
            ref_area,
            color_profiles,
            is_white=is_white,
            ref_shape=ref_shape,
        )

        # --- Encode annotated image ---
        annotated_b64 = encode_image_to_base64(annotated_image)

        return jsonify({
            "count": count,
            "annotated_image": annotated_b64,
            "ref_area_px": round(ref_area, 1),
            "is_white_pill": is_white,
            "num_color_clusters": len(color_profiles),
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    except Exception:
        app.logger.error("Unexpected error during analysis:\n" + traceback.format_exc())
        return jsonify({
            "error": "An unexpected error occurred during image processing. "
                     "Please try again with clearer photos."
        }), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
