from pathlib import Path

import pandas as pd

from config import CLASS_NAMES, CSV_COLUMNS, TRAIN_CSV, VAL_CSV
from preprocessing import preprocess_text


def read_twitter_csv(path):
    path = Path(path)
    return pd.read_csv(path, header=None, names=CSV_COLUMNS, encoding="utf-8")


def clean_twitter_dataframe(df):
    cleaned = df.copy()
    cleaned["text"] = cleaned["text"].map(preprocess_text)
    cleaned["sentiment"] = cleaned["sentiment"].map(preprocess_text)
    cleaned = cleaned.dropna(subset=["text", "sentiment"])
    cleaned = cleaned[cleaned["sentiment"].isin(CLASS_NAMES)]
    return cleaned.reset_index(drop=True)


def load_twitter_csv(path):
    return clean_twitter_dataframe(read_twitter_csv(path))


def load_train_val(train_path=TRAIN_CSV, val_path=VAL_CSV):
    return load_twitter_csv(train_path), load_twitter_csv(val_path)
