import io

import pytest
from PIL import Image

from app import create_app


class FakeCaptioner:
    def __init__(self):
        self.images = []

    def describe(self, image):
        self.images.append(image)
        return "um gato sentado no sofá"


@pytest.fixture
def captioner():
    return FakeCaptioner()


@pytest.fixture
def client(captioner):
    app = create_app(captioner)
    app.config["TESTING"] = True
    return app.test_client()


def image_file(mode="RGB", size=(10, 10), fmt="PNG"):
    buffer = io.BytesIO()
    Image.new(mode, size).save(buffer, format=fmt)
    buffer.seek(0)
    return buffer


def test_descreve_imagem(client, captioner):
    res = client.post("/analyze-image", data={"image": (image_file(), "foto.png")})
    assert res.status_code == 200
    assert res.get_json() == {"description": "um gato sentado no sofá"}
    assert captioner.images[0].mode == "RGB"


def test_converte_imagem_com_transparencia(client, captioner):
    res = client.post("/analyze-image", data={"image": (image_file("RGBA"), "foto.png")})
    assert res.status_code == 200
    assert captioner.images[0].mode == "RGB"


def test_sem_arquivo(client):
    res = client.post("/analyze-image", data={})
    assert res.status_code == 400


def test_arquivo_que_nao_e_imagem(client, captioner):
    res = client.post("/analyze-image", data={"image": (io.BytesIO(b"texto"), "nota.txt")})
    assert res.status_code == 400
    assert captioner.images == []


def test_imagem_maior_que_o_limite(client, monkeypatch):
    client.application.config["MAX_CONTENT_LENGTH"] = 1024
    res = client.post("/analyze-image", data={"image": (io.BytesIO(b"0" * 4096), "grande.png")})
    assert res.status_code == 413


def test_serve_frontend(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"<title>" in res.data
