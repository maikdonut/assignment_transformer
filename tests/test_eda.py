from pathlib import Path

import pandas as pd

import eda


def _write_dataset(path):
    rows = [
        [1, "Game", "Irrelevant", "off topic"],
        [2, "Game", "Negative", "bad game"],
        [3, "Movie", "Neutral", "release today"],
        [4, "Movie", "Positive", "great movie"],
        [5, "Movie", "Positive", None],
    ]
    pd.DataFrame(rows).to_csv(path, header=False, index=False)


def test_run_eda_saves_report_and_plots(tmp_path, monkeypatch):
    train_path = Path(tmp_path) / "train.csv"
    val_path = Path(tmp_path) / "val.csv"
    _write_dataset(train_path)
    _write_dataset(val_path)
    monkeypatch.setattr(eda, "TRAIN_CSV", train_path)
    monkeypatch.setattr(eda, "VAL_CSV", val_path)
    monkeypatch.setattr(eda, "EDA_REPORT_PATH", Path(tmp_path) / "eda.txt")
    monkeypatch.setattr(eda, "EDA_PLOTS_DIR", Path(tmp_path) / "plots")

    report = eda.run_eda()

    assert "Пропуски до очистки" in report
    assert "Распределение классов" in report
    assert "ПЕРВЫЕ ВЫВОДЫ" in report
    assert eda.EDA_REPORT_PATH.exists()
    assert (eda.EDA_PLOTS_DIR / "class_distribution.png").exists()
