import sys

import pandas as pd

from config import CLASS_NAMES, ERROR_ANALYSIS_PATH, VAL_CSV
from data import load_train_val
from inference import load_sentiment_model, predict_sentiments

POSITIVE = "Positive"
NEGATIVE = "Negative"


def _console(text):
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(text).encode(encoding, errors="replace").decode(encoding)


def _format_examples(df, limit=None):
    rows = df if limit is None else df.head(limit)
    blocks = []
    for _, row in rows.iterrows():
        blocks.append(
            f"\nТекст: {row['text']}\n"
            f"Entity: {row['entity']}\n"
            f"Истинный: {row['true_label']}, Предсказан: {row['pred_label']}"
        )
    return "".join(blocks) if blocks else "\n(нет примеров)\n"


def _confusion_pairs(errors):
    counts = (
        errors.groupby(["true_label", "pred_label"])
        .size()
        .reindex(
            pd.MultiIndex.from_product(
                [CLASS_NAMES, CLASS_NAMES], names=["true_label", "pred_label"]
            ),
            fill_value=0,
        )
    )
    lines = []
    for (true_label, pred_label), count in counts.items():
        if true_label == pred_label or count == 0:
            continue
        lines.append(f"  {true_label} → {pred_label}: {int(count)}")
    return "\n".join(lines) if lines else "  (нет ошибок)"


def _entity_counts(errors, top_n=10):
    counts = errors["entity"].value_counts().head(top_n)
    if counts.empty:
        return "  (нет ошибок)"
    return "\n".join(f"  {name}: {int(n)}" for name, n in counts.items())


def categorize_positive_errors(errors):
    """Разделяет multiclass-ошибки по схеме Positive vs Rest."""
    fp = errors[
        (errors["pred_label"] == POSITIVE) & (errors["true_label"] != POSITIVE)
    ]
    fn = errors[
        (errors["true_label"] == POSITIVE) & (errors["pred_label"] != POSITIVE)
    ]
    other = errors.drop(index=fp.index.union(fn.index))
    return fp, fn, other


def generate_observations(df_test, errors, fp, fn):
    n_test = len(df_test)
    n_errors = len(errors)
    error_rate = n_errors / n_test * 100 if n_test else 0.0
    mean_all = df_test["text"].str.len().mean() if n_test else 0.0
    mean_errors = errors["text"].str.len().mean() if n_errors else 0.0

    pairs = (
        errors.groupby(["true_label", "pred_label"])
        .size()
        .sort_values(ascending=False)
        .head(3)
    )
    pair_text = (
        ", ".join(f"{true} → {pred}: {int(count)}" for (true, pred), count in pairs.items())
        if not pairs.empty
        else "ошибок нет"
    )
    url_count = int(errors["text"].str.contains(r"https?://|www\.|t\.co/", case=False, regex=True).sum())
    hashtag_count = int(errors["text"].str.contains("#", regex=False).sum())
    short_count = int((errors["text"].str.len() < 30).sum())
    top_entity = errors["entity"].value_counts().head(1)
    entity_text = (
        f"{top_entity.index[0]} ({int(top_entity.iloc[0])})"
        if not top_entity.empty
        else "нет"
    )
    negative_to_positive = int(
        (
            (errors["true_label"] == NEGATIVE)
            & (errors["pred_label"] == POSITIVE)
        ).sum()
    )
    positive_to_negative = int(
        (
            (errors["true_label"] == POSITIVE)
            & (errors["pred_label"] == NEGATIVE)
        ).sum()
    )

    length_relation = "длиннее" if mean_errors > mean_all else "короче или равны"
    return (
        f"Отчёт построен автоматически по текущим предсказаниям.\n"
        f"- Ошибок: {n_errors}/{n_test} ({error_rate:.1f}%).\n"
        f"- Positive vs Rest: False Positives — {len(fp)}, "
        f"False Negatives — {len(fn)}.\n"
        f"- Прямая смена полярности: Negative → Positive — "
        f"{negative_to_positive}, Positive → Negative — {positive_to_negative}.\n"
        f"- Главные пары путаницы: {pair_text}.\n"
        f"- Ошибочные тексты в среднем {length_relation} полного набора: "
        f"{mean_errors:.0f} против {mean_all:.0f} символов.\n"
        f"- Среди ошибок: URL — {url_count}, хештег — {hashtag_count}, "
        f"коротких текстов (<30 символов) — {short_count}.\n"
        f"- Entity с наибольшим числом ошибок: {entity_text}."
    )


def write_error_analysis(df_test, errors, fp, fn, other, observations):
    n_test = len(df_test)
    n_errors = len(errors)
    mean_err_len = errors["text_length"].mean() if n_errors else 0.0
    mean_all_len = df_test["text"].str.len().mean()

    report = (
        "=== АНАЛИЗ ОШИБОК ===\n\n"
        f"split: val={VAL_CSV.name} ({n_test})\n"
        f"Всего ошибок: {n_errors} ({n_errors / n_test * 100:.1f}%)\n"
        f"False Positives (предсказан Positive, true не Positive): {len(fp)}\n"
        f"False Negatives (true Positive, предсказан не Positive): {len(fn)}\n"
        f"Прочие ошибки: {len(other)}\n\n"
        "=== ПАРЫ ПУТАНИЦЫ (true → pred) ===\n"
        f"{_confusion_pairs(errors)}\n\n"
        "=== ДЛИНА ТЕКСТОВ ===\n"
        f"Средняя длина ошибочных текстов: {mean_err_len:.0f}\n"
        f"Средняя длина всех текстов: {mean_all_len:.0f}\n\n"
        "=== ТОП ENTITY СРЕДИ ОШИБОК ===\n"
        f"{_entity_counts(errors)}\n\n"
        "=== ПРИМЕРЫ FALSE POSITIVES ===\n"
        "предсказала Positive, истинный класс не Positive\n"
        f"{_format_examples(fp)}\n\n"
        "=== ПРИМЕРЫ FALSE NEGATIVES ===\n"
        "истинный класс Positive, предсказан другой класс\n"
        f"{_format_examples(fn)}\n\n"
        "=== ПРОЧИЕ ОШИБКИ ===\n"
        f"{_format_examples(other)}\n\n"
        "=== НАБЛЮДЕНИЯ ===\n"
        f"{observations}\n"
    )
    ERROR_ANALYSIS_PATH.write_text(report, encoding="utf-8")
    print(f"Анализ сохранён: {ERROR_ANALYSIS_PATH}")
    return report


def run_error_analysis(observations=None):
    model = load_sentiment_model()

    _, val_df = load_train_val()
    test_texts = val_df["text"].tolist()
    print(f"Val: {len(test_texts)} примеров ({VAL_CSV.name})")

    preds = predict_sentiments(test_texts, model)
    df_test = pd.DataFrame(
        {
            "text": test_texts,
            "true_label": val_df["sentiment"].astype(str).tolist(),
            "pred_label": [p["prediction"] for p in preds],
            "entity": val_df["entity"].astype(str).tolist(),
        }
    )

    errors = df_test[df_test["true_label"] != df_test["pred_label"]].copy()
    errors["text_length"] = errors["text"].str.len()

    fp, fn, other = categorize_positive_errors(errors)

    print(f"Всего ошибок: {len(errors)}")
    print(f"False Positives: {len(fp)}")
    print(f"False Negatives: {len(fn)}")
    print(f"Прочие: {len(other)}")
    print("\n=== FALSE POSITIVES (Positive вместо любого другого класса) ===")
    for _, row in fp.iterrows():
        print(_console(f"\nТекст: {row['text'][:100]}..."))
        print(f"Истинный класс: {row['true_label']}, Предсказан: {row['pred_label']}")
    print("\n=== FALSE NEGATIVES (Positive предсказан другим классом) ===")
    for _, row in fn.iterrows():
        print(_console(f"\nТекст: {row['text'][:100]}..."))
        print(f"Истинный класс: {row['true_label']}, Предсказан: {row['pred_label']}")
    print(f"\nСредняя длина ошибочных текстов: {errors['text_length'].mean():.0f}")
    print(f"Средняя длина всех текстов: {df_test['text'].str.len().mean():.0f}")

    if observations is None:
        observations = generate_observations(df_test, errors, fp, fn)

    write_error_analysis(df_test, errors, fp, fn, other, observations)
    return df_test, errors, fp, fn, other


if __name__ == "__main__":
    run_error_analysis()
