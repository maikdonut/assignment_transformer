from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    BATCH_SIZE,
    FINE_TUNED_MODEL_DIR,
    MAX_LENGTH,
)
from embeddings import get_device
from preprocessing import preprocess_texts


@dataclass
class SentimentModel:
    tokenizer: object
    model: object
    device: torch.device
    id2label: dict
    source: str


REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def validate_model_dir(model_dir=FINE_TUNED_MODEL_DIR):
    model_dir = Path(model_dir)
    missing = [name for name in REQUIRED_MODEL_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Локальный checkpoint неполный: {model_dir}. "
            f"Отсутствуют: {', '.join(missing)}. "
            "После clone выполните: git lfs install && git lfs pull"
        )

    weights_path = model_dir / "model.safetensors"
    with weights_path.open("rb") as file:
        prefix = file.read(64)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            "Вместо model.safetensors найден Git LFS pointer. "
            "Выполните: git lfs install && git lfs pull"
        )
    return model_dir


def load_sentiment_model(model_dir=None, device=None):
    source = validate_model_dir(model_dir or FINE_TUNED_MODEL_DIR)
    device = device or get_device()
    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSequenceClassification.from_pretrained(source)
    except OSError as exc:
        raise RuntimeError(
            f"Не удалось загрузить локальный checkpoint '{source}'. "
            "Проверьте файлы модели или повторите: git lfs pull"
        ) from exc
    model.to(device)
    model.eval()
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    print(f"Модель загружена: {source} (device={device})")
    return SentimentModel(tokenizer, model, device, id2label, str(source))


@lru_cache(maxsize=1)
def get_sentiment_model():
    return load_sentiment_model()


def predict_sentiments(texts, sentiment_model=None, batch_size=BATCH_SIZE):
    if isinstance(texts, str):
        texts = [texts]
    prepared = preprocess_texts(list(texts))
    loaded = sentiment_model or get_sentiment_model()
    predictions = []

    for start in range(0, len(prepared), batch_size):
        batch_texts = prepared[start : start + batch_size]
        inputs = loaded.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {key: value.to(loaded.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = loaded.model(**inputs).logits
        probabilities = F.softmax(logits, dim=1)
        prediction_ids = torch.argmax(probabilities, dim=1)

        for index, text in enumerate(batch_texts):
            label_id = int(prediction_ids[index].item())
            predictions.append(
                {
                    "text": text,
                    "prediction": loaded.id2label[label_id],
                    "label_id": label_id,
                    "probabilities": probabilities[index].cpu().numpy(),
                }
            )
    return predictions
