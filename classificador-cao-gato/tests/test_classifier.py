import io

import pytest
import torch
from PIL import Image

import classifier


class FixedModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.logits = torch.tensor(logits)

    def forward(self, batch):
        assert batch.shape == (1, 3, 224, 224)
        return self.logits.repeat(batch.shape[0], 1)


def png_bytes(mode="RGB", size=(50, 80)):
    buffer = io.BytesIO()
    Image.new(mode, size, "white").save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


@pytest.mark.parametrize("logits, label", [([5.0, 0.0], "Gato"), ([0.0, 5.0], "Cachorro")])
def test_predict_retorna_classe_confiante(logits, label):
    result = classifier.predict(FixedModel(logits), png_bytes())
    assert result.label == label
    assert result.is_confident
    assert result.confidence > 0.9


def test_predict_sem_confianca_quando_abaixo_do_limite():
    result = classifier.predict(FixedModel([0.1, 0.0]), png_bytes())
    assert result.label is None
    assert not result.is_confident


def test_predict_aceita_imagem_com_transparencia_e_tons_de_cinza():
    for mode in ("RGBA", "L"):
        assert classifier.predict(FixedModel([5.0, 0.0]), png_bytes(mode)).label == "Gato"


def test_modelo_salvo_carrega_com_mesma_arquitetura(tmp_path):
    path = tmp_path / "modelo.pth"
    torch.save(classifier.build_model().state_dict(), path)
    model = classifier.load_model(path, torch.device("cpu"))
    assert not model.training
    assert model.fc.out_features == 2


def test_download_nao_baixa_se_arquivo_existe(tmp_path, monkeypatch):
    path = tmp_path / "modelo.pth"
    path.write_bytes(b"x")
    monkeypatch.setattr(classifier.urllib.request, "urlretrieve", lambda *a: pytest.fail("não deveria baixar"))
    assert classifier.download_model(path) == path
