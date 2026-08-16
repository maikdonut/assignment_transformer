from transformers import AutoTokenizer

from config import EN_MODEL_NAME, MAX_LENGTH
from preprocessing import preprocess_text, preprocess_texts


def load_tokenizer(model_name=EN_MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_texts(texts, max_length=MAX_LENGTH, tokenizer=None):
    if tokenizer is None:
        tokenizer = load_tokenizer()
    texts = preprocess_texts(texts)
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def explain_tokenization(text, tokenizer=None):
    if tokenizer is None:
        tokenizer = load_tokenizer()
    prepared = preprocess_text(text)
    if prepared is None:
        raise ValueError("Текст не должен быть пустым")

    tokens = tokenizer.tokenize(prepared)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    explanation = {
        "text": prepared,
        "tokens": tokens,
        "ids": ids,
        "count": len(tokens),
    }
    print(f"Исходный текст: {prepared}")
    print(f"Токены: {tokens}")
    print(f"IDs: {ids}")
    print(f"Количество: {len(tokens)}")
    return explanation
