from transformers import AutoTokenizer

from config import EN_MODEL_NAME, MAX_LENGTH


def load_tokenizer(model_name=EN_MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_texts(texts, max_length=MAX_LENGTH, tokenizer=None):
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
