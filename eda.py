import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import (
    CLASS_NAMES,
    EDA_PLOTS_DIR,
    EDA_REPORT_PATH,
    TRAIN_CSV,
    VAL_CSV,
)
from data import clean_twitter_dataframe, read_twitter_csv


def _console(text):
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(text).encode(encoding, errors="replace").decode(encoding)


def _dataset_summary(name, raw_df, clean_df):
    lengths = clean_df["text"].str.len()
    class_counts = clean_df["sentiment"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    examples = []
    for label in CLASS_NAMES:
        rows = clean_df.loc[clean_df["sentiment"] == label, ["entity", "text"]].head(2)
        examples.append(f"\n{label}:")
        for _, row in rows.iterrows():
            examples.append(f"  [{row['entity']}] {row['text']}")

    return (
        f"=== {name} ===\n"
        f"Сырые строки: {len(raw_df)}\n"
        f"Строки после preprocessing: {len(clean_df)}\n"
        f"Удалено: {len(raw_df) - len(clean_df)}\n"
        f"Типы колонок:\n{raw_df.dtypes.to_string()}\n"
        f"Пропуски до очистки:\n{raw_df.isna().sum().to_string()}\n"
        f"Дубликаты текстов: {int(clean_df['text'].duplicated().sum())}\n"
        f"Распределение классов:\n{class_counts.to_string()}\n"
        f"Доли классов:\n{(class_counts / len(clean_df) * 100).round(2).to_string()}\n"
        f"Длины текстов:\n{lengths.describe().round(2).to_string()}\n"
        f"Уникальных entity: {clean_df['entity'].nunique()}\n"
        f"Топ entity:\n{clean_df['entity'].value_counts().head(10).to_string()}\n"
        f"Примеры по классам:{''.join(examples)}\n"
    )


def _save_plots(train_df, val_df):
    EDA_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    class_frame = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="validation"),
        ],
        ignore_index=True,
    )
    plt.figure(figsize=(9, 5))
    sns.countplot(
        data=class_frame,
        x="sentiment",
        hue="split",
        order=CLASS_NAMES,
    )
    plt.title("Распределение классов")
    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR / "class_distribution.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(train_df["text"].str.len(), bins=50)
    plt.title("Распределение длины train-текстов")
    plt.xlabel("Количество символов")
    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR / "text_lengths.png")
    plt.close()


def run_eda():
    raw_train = read_twitter_csv(TRAIN_CSV)
    raw_val = read_twitter_csv(VAL_CSV)
    train_df = clean_twitter_dataframe(raw_train)
    val_df = clean_twitter_dataframe(raw_val)

    train_counts = train_df["sentiment"].value_counts(normalize=True)
    largest_class = train_counts.idxmax()
    shortest = train_df["text"].str.len().median()
    report = (
        "EDA: Twitter Entity Sentiment\n\n"
        f"{_dataset_summary('TRAIN', raw_train, train_df)}\n"
        f"{_dataset_summary('VALIDATION', raw_val, val_df)}\n"
        "=== ПЕРВЫЕ ВЫВОДЫ ===\n"
        f"- В train крупнейший класс — {largest_class} "
        f"({train_counts[largest_class] * 100:.1f}%).\n"
        f"- Медианная длина train-текста — {shortest:.0f} символов.\n"
        f"- После preprocessing осталось {len(train_df)}/{len(raw_train)} train "
        f"и {len(val_df)}/{len(raw_val)} validation примеров.\n"
        "- Для итоговой оценки используется отдельный validation-файл; "
        "подбор baseline выполняется только на train.\n"
    )
    EDA_REPORT_PATH.write_text(report, encoding="utf-8")
    _save_plots(train_df, val_df)
    print(_console(report))
    print(f"EDA сохранён: {EDA_REPORT_PATH}, графики: {EDA_PLOTS_DIR}")
    return report


if __name__ == "__main__":
    run_eda()
