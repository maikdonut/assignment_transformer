from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from error_analysis import generate_observations
from config import FINE_TUNED_MODEL_ID
from inference import SentimentModel, predict_sentiments, resolve_model_source


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return {
            "input_ids": torch.ones((len(texts), 3), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), 3), dtype=torch.long),
        }


class FakeModel:
    def __call__(self, **kwargs):
        rows = kwargs["input_ids"].shape[0]
        logits = torch.tensor([[0.0, 0.0, 0.0, 4.0]]).repeat(rows, 1)
        return SimpleNamespace(logits=logits)


def test_predict_sentiments_uses_common_preprocessing():
    loaded = SentimentModel(
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        device=torch.device("cpu"),
        id2label={0: "Irrelevant", 1: "Negative", 2: "Neutral", 3: "Positive"},
        source="fake",
    )

    result = predict_sentiments(["  Great!  "], loaded)[0]

    assert result["text"] == "Great!"
    assert result["prediction"] == "Positive"
    assert np.isclose(result["probabilities"].sum(), 1.0)


def test_model_source_prefers_local_then_hub(tmp_path):
    missing = tmp_path / "missing"
    local = tmp_path / "model"
    local.mkdir()

    assert resolve_model_source(missing) == FINE_TUNED_MODEL_ID
    assert resolve_model_source(local) == str(local)


def test_generate_observations_uses_current_counts():
    df_test = pd.DataFrame(
        {
            "text": ["good", "bad https://x.test", "#neutral"],
            "true_label": ["Positive", "Negative", "Neutral"],
            "pred_label": ["Negative", "Positive", "Neutral"],
            "entity": ["A", "A", "B"],
        }
    )
    errors = df_test.iloc[:2].copy()
    fp = errors.iloc[[1]]
    fn = errors.iloc[[0]]

    observations = generate_observations(df_test, errors, fp, fn)

    assert "2/3 (66.7%)" in observations
    assert "FP Positive вместо Negative — 1" in observations
    assert "FN Negative вместо Positive — 1" in observations
    assert "URL — 1" in observations
