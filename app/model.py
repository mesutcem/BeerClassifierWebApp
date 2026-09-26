from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CLASS_NAMES = ("IPA", "Light Lager", "Premium Lager")


class BeerClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_stack(x)


def find_model_path() -> Path:
    configured_path = os.getenv("MODEL_PATH")
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = BASE_DIR / path
        return path

    preferred_path = BASE_DIR / "beer_classifier.pth"
    if preferred_path.exists():
        return preferred_path

    candidates = sorted(BASE_DIR.glob("*.pth"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "Model bulunamadı. .pth dosyasını proje köküne ekleyin veya "
            "MODEL_PATH ortam değişkenini ayarlayın."
        )
    raise RuntimeError(
        "Birden fazla .pth dosyası bulundu. Kullanılacak dosyayı MODEL_PATH ile belirtin."
    )


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint bir state_dict veya checkpoint sözlüğü olmalı.")

    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint

    raise ValueError("Checkpoint içinde model ağırlıkları bulunamadı.")


def load_model() -> tuple[BeerClassifier, Path]:
    model_path = find_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    # weights_only=True, pickle tabanlı keyfi nesne yüklemeyi engeller.
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = _extract_state_dict(checkpoint)

    # DataParallel ile kaydedilmiş checkpoint'lerdeki "module." önekini temizle.
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }

    model = BeerClassifier()
    model.load_state_dict(state_dict)
    model.eval()
    return model, model_path


def get_class_names() -> tuple[str, str, str]:
    configured_names = os.getenv("CLASS_NAMES")
    if not configured_names:
        return DEFAULT_CLASS_NAMES

    names = tuple(name.strip() for name in configured_names.split(",") if name.strip())
    if len(names) != 3:
        raise ValueError("CLASS_NAMES virgülle ayrılmış tam olarak 3 sınıf içermeli.")
    return names  # type: ignore[return-value]


@torch.inference_mode()
def predict(
    model: BeerClassifier,
    features: list[float],
    class_names: tuple[str, str, str],
) -> dict[str, Any]:
    inputs = torch.tensor([features], dtype=torch.float32)
    logits = model(inputs)
    probabilities = torch.softmax(logits, dim=1)[0]
    predicted_index = int(torch.argmax(probabilities).item())

    return {
        "class_index": predicted_index,
        "class_name": class_names[predicted_index],
        "confidence": round(float(probabilities[predicted_index].item()), 6),
        "probabilities": {
            name: round(float(probability.item()), 6)
            for name, probability in zip(class_names, probabilities, strict=True)
        },
    }
