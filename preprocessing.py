import math
import re


_WHITESPACE_RE = re.compile(r"\s+")


def preprocess_text(value):
    """Нормализует текст, не удаляя sentiment-сигналы."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    text = _WHITESPACE_RE.sub(" ", str(value)).strip()
    return text or None


def preprocess_texts(texts):
    """Готовит батч; невалидный inference-ввод считается ошибкой."""
    prepared = []
    for value in texts:
        text = preprocess_text(value)
        if text is None:
            raise ValueError("Текст не должен быть пустым")
        prepared.append(text)
    return prepared
