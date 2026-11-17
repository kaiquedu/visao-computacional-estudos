from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Union

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

CLASSES = ("Gato", "Cachorro")
CONFIDENCE_THRESHOLD = 0.7
MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "modelo_dogs_vs_cats.pth"))
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/kaiquedu/visao-computacional-estudos/releases/download/modelo-v1/modelo_dogs_vs_cats.pth",
)

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


@dataclass(frozen=True)
class Prediction:
    label: str | None
    confidence: float

    @property
    def is_confident(self) -> bool:
        return self.label is not None


def build_model(pretrained: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def download_model(path: Path = MODEL_PATH, url: str = MODEL_URL) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


def load_model(path: Path = MODEL_PATH, device: torch.device | None = None) -> nn.Module:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    model.load_state_dict(torch.load(download_model(path), map_location=device, weights_only=True))
    return model.to(device).eval()


def predict(model: nn.Module, image: Union[Image.Image, BinaryIO]) -> Prediction:
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    device = next(model.parameters()).device
    batch = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(batch)[0], dim=0)
    confidence, index = torch.max(probabilities, 0)
    label = CLASSES[index.item()] if confidence.item() >= CONFIDENCE_THRESHOLD else None
    return Prediction(label=label, confidence=confidence.item())
