from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    BATCH_SIZE,
    FINE_TUNED_MODEL_DIR,
    FINE_TUNED_MODEL_ID,
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


def resolve_model_source(local_dir=FINE_TUNED_MODEL_DIR):
    local_dir = Path(local_dir)
    return str(local_dir) if local_dir.exists() else FINE_TUNED_MODEL_ID


def load_sentiment_model(source=None, device=None):
    source = source or resolve_model_source()
    device = device or get_device()
    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSequenceClassification.from_pretrained(source)
    except OSError as exc:
        raise RuntimeError(
            f"Не удалось загрузить модель '{source}'. Для offline-запуска "
            f"поместите checkpoint в {FINE_TUNED_MODEL_DIR}."
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
