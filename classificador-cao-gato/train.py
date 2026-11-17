import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from classifier import CLASSES, build_model, preprocess


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_split(root: Path) -> datasets.ImageFolder:
    dataset = datasets.ImageFolder(root, transform=preprocess, is_valid_file=is_valid_image)
    expected = [c.lower() for c in ("cats", "dogs")]
    if [c.lower() for c in dataset.classes] != expected:
        raise SystemExit(f"{root} deve conter as pastas cats/ e dogs/, encontrado: {dataset.classes}")
    return dataset


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predicted = model(inputs).argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o classificador de cachorros e gatos")
    parser.add_argument("--data", type=Path, default=Path("data"), help="pasta com train/ e test/")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("modelo_dogs_vs_cats.pth"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(load_split(args.data / "train"), batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(load_split(args.data / "test"), batch_size=args.batch_size, num_workers=2)

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start = time.time()
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Época {epoch + 1}/{args.epochs}, Loss: {loss:.4f}")
    print(f"Treinamento concluído em {time.time() - start:.2f} segundos")

    print(f"Acurácia no conjunto de teste: {evaluate(model, test_loader, device):.2f}%")
    torch.save(model.state_dict(), args.output)
    print(f"Modelo salvo em {args.output} com classes {CLASSES}")


if __name__ == "__main__":
    main()
