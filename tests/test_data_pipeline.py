from pathlib import Path

import pandas as pd

from config import CSV_COLUMNS
from data import load_twitter_csv
from embeddings import cache_fingerprint
from preprocessing import preprocess_text, preprocess_texts
from tokenization import explain_tokenization


class FakeTokenizer:
    def tokenize(self, text):
        return text.lower().split()

    def convert_tokens_to_ids(self, tokens):
        return list(range(1, len(tokens) + 1))


def test_preprocess_text_preserves_sentiment_content():
    assert preprocess_text("  Great!  😄\nhttps://x.test  ") == "Great! 😄 https://x.test"
    assert preprocess_text(" \n\t ") is None
    assert preprocess_text(None) is None
    assert preprocess_texts([" a ", "b"]) == ["a", "b"]


def test_load_twitter_csv_filters_invalid_rows(tmp_path):
    rows = [
        [1, "Game", "Positive", "  great  "],
        [2, "Game", "Unknown", "text"],
        [3, "Game", "Negative", "   "],
    ]
    path = Path(tmp_path) / "sample.csv"
    pd.DataFrame(rows).to_csv(path, header=False, index=False)

    result = load_twitter_csv(path)

    assert result[CSV_COLUMNS].to_dict("records") == [
        {"tweet_id": 1, "entity": "Game", "sentiment": "Positive", "text": "great"}
    ]


def test_explain_tokenization_returns_reusable_result():
    result = explain_tokenization(" Transformers   are amazing! ", FakeTokenizer())
    assert result["tokens"] == ["transformers", "are", "amazing!"]
    assert result["ids"] == [1, 2, 3]
    assert result["count"] == 3


def test_cache_fingerprint_changes_with_text_or_settings():
    base = cache_fingerprint(["one", "two"])
    assert base == cache_fingerprint(["one", "two"])
    assert base != cache_fingerprint(["one", "changed"])
    assert base != cache_fingerprint(["one", "two"], max_length=64)
