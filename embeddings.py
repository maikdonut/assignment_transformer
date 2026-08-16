import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from config import BATCH_SIZE, EN_MODEL_NAME, MAX_LENGTH
from tokenization import load_tokenizer, tokenize_texts


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(model_name=EN_MODEL_NAME, device=None, output_attentions=False):
    if device is None:
        device = get_device()
    tokenizer = load_tokenizer(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=output_attentions)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def cache_fingerprint(texts, model_name=EN_MODEL_NAME, max_length=MAX_LENGTH):
    digest = hashlib.sha256()
    digest.update(f"{model_name}\0{max_length}\0".encode())
    for text in texts:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _cache_metadata_path(cache_path):
    return cache_path.with_suffix(cache_path.suffix + ".json")


def _load_valid_cache(cache_path, fingerprint, expected_rows):
    metadata_path = _cache_metadata_path(cache_path)
    if not cache_path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != fingerprint:
            return None
        embeddings = np.load(cache_path)
        if embeddings.shape[0] != expected_rows:
            return None
        return embeddings
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def get_cls_embeddings(
    texts,
    batch_size=BATCH_SIZE,
    tokenizer=None,
    model=None,
    device=None,
    max_length=MAX_LENGTH,
    cache_path=None,
):
    cache_path = Path(cache_path) if cache_path is not None else None
    fingerprint = cache_fingerprint(texts, max_length=max_length)
    if cache_path is not None:
        cached = _load_valid_cache(cache_path, fingerprint, len(texts))
        if cached is not None:
            print(f"Загружаю эмбеддинги из кэша: {cache_path}")
            return cached
        if cache_path.exists():
            print(f"Кэш устарел, пересчитываю: {cache_path}")

    if tokenizer is None or model is None:
        tokenizer, model, device = load_encoder(device=device)
    elif device is None:
        device = next(model.parameters()).device

    all_embeddings = []
    n_texts = len(texts)
    for start in range(0, n_texts, batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenize_texts(batch_texts, max_length=max_length, tokenizer=tokenizer)
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            outputs = model(**tokens)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())
        done = min(start + batch_size, n_texts)
        if done == n_texts or done % (batch_size * 20) == 0:
            print(f"Эмбеддинги: {done}/{n_texts}")

    embeddings = np.vstack(all_embeddings)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        metadata = {
            "fingerprint": fingerprint,
            "rows": len(texts),
            "model": EN_MODEL_NAME,
            "max_length": max_length,
        }
        _cache_metadata_path(cache_path).write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        print(f"Сохранены эмбеддинги: {cache_path}")

    return embeddings
