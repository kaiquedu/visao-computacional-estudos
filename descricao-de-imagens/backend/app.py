from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from captioner import Captioner

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
MAX_PIXELS = 4096 * 4096


def create_app(captioner: Captioner) -> Flask:
    app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    app.json.ensure_ascii = False

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error=f"A imagem deve ter no máximo {MAX_UPLOAD_MB} MB."), 413

    @app.get("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/analyze-image")
    def analyze_image():
        file = request.files.get("image")
        if file is None or not file.filename:
            return jsonify(error="Nenhum arquivo de imagem enviado."), 400
        try:
            image = Image.open(file.stream)
            if image.width * image.height > MAX_PIXELS:
                return jsonify(error="Resolução da imagem muito alta."), 400
            image = image.convert("RGB")
        except (UnidentifiedImageError, OSError):
            return jsonify(error="Arquivo enviado não é uma imagem válida."), 400

        return jsonify(description=captioner.describe(image))

    return app


if __name__ == "__main__":
    from captioner import BlipCaptioner

    create_app(BlipCaptioner()).run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")))
