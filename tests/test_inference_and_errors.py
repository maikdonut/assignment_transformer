from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from error_analysis import categorize_positive_errors, generate_observations
from inference import (
    REQUIRED_MODEL_FILES,
    SentimentModel,
    predict_sentiments,
    validate_model_dir,
)


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


def test_model_dir_is_required_and_never_falls_back_to_hub(tmp_path):
    with pytest.raises(FileNotFoundError, match="git lfs pull"):
        validate_model_dir(tmp_path / "missing")

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in REQUIRED_MODEL_FILES:
        (model_dir / name).write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_text(
        "version https://git-lfs.github.com/spec/v1\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        validate_model_dir(model_dir)


def test_positive_fp_fn_are_multiclass_one_vs_rest():
    errors = pd.DataFrame(
        {
            "text": [
                "negative to positive",
                "neutral to positive",
                "irrelevant to positive",
                "positive to negative",
                "positive to neutral",
                "positive to irrelevant",
                "negative to neutral",
            ],
            "true_label": [
                "Negative",
                "Neutral",
                "Irrelevant",
                "Positive",
                "Positive",
                "Positive",
                "Negative",
            ],
            "pred_label": [
                "Positive",
                "Positive",
                "Positive",
                "Negative",
                "Neutral",
                "Irrelevant",
                "Neutral",
            ],
            "entity": ["A"] * 7,
        }
    )
    fp, fn, other = categorize_positive_errors(errors)

    assert len(fp) == 3
    assert len(fn) == 3
    assert len(other) == 1

    observations = generate_observations(errors, errors, fp, fn)

    assert "False Positives — 3" in observations
    assert "False Negatives — 3" in observations
    assert "Negative → Positive — 1" in observations
    assert "Positive → Negative — 1" in observations
