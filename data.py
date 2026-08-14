from pathlib import Path

import pandas as pd

from config import CSV_COLUMNS, TRAIN_CSV, VAL_CSV


def load_twitter_csv(path):
    path = Path(path)
    df = pd.read_csv(path, header=None, names=CSV_COLUMNS, encoding="utf-8")
    df = df.dropna(subset=["text", "sentiment"])
    df["text"] = df["text"].astype(str)
    df["sentiment"] = df["sentiment"].astype(str)
    return df.reset_index(drop=True)


def load_train_val(train_path=TRAIN_CSV, val_path=VAL_CSV):
    return load_twitter_csv(train_path), load_twitter_csv(val_path)
