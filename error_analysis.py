import sys

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from compare import CLASS_NAMES, predict_fine_tuned
from config import ERROR_ANALYSIS_PATH, FINE_TUNED_MODEL_DIR, VAL_CSV
from data import load_train_val
from embeddings import get_device

POSITIVE = "Positive"
NEGATIVE = "Negative"

OBSERVATIONS = """\
Fine-tuned DistilBERT ошибается редко (38/1000, 3.8%) и почти не путает полярность:
Positive ↔ Negative случилось один раз, False Negatives нет. Модель уверенно отличает
«хорошо» от «плохо».

Большинство ошибок — соседние классы, а не противоположные:
- Neutral ↔ Positive (11) и Neutral ↔ Irrelevant (8) — главные пары;
- Negative чаще уходит в Neutral (7) или Irrelevant (4), чем в Positive (1).
Граница Neutral / Irrelevant в Twitter Entity Sentiment размыта: твит может быть
про игру, новостной ссылкой или оффтопом с хештегом бренда.

Ошибочные тексты длиннее средних (158 vs 132 символа). В длинных твитах смешаны
несколько тем, новости, URL и оговорки («игра сырая, но атмосфера невероятная»),
из-за чего золотая метка и поверхностный тон расходятся.

Типичные паттерны:
1. Двусмысленная лексика. Единственный FP — «I'm addicted to call of duty mobile😅»:
   «addicted» в разметке Negative, для модели это скорее энтузиазм.
2. Entity только в хештеге / мимоходом. Жильё «beyond the call of duty», листинг
   Poshmark с #leagueoflegends, закладка сестры с #RainbowSixSiege — в разметке
   Irrelevant/Neutral, модель цепляется за имя сущности или позитивные слова.
3. Новости и ссылки. Мемо Facebook, подкаст Johnson & Johnson, бестселлер Amazon
   размечены как Negative, предсказание Neutral: тон фактологический, без явной брани.
4. Смешанная оценка. Ghost Recon «нужны фиксы, но виды потрясающие», сравнение
   Xbox/PS5, ностальгия по картам Rainbow Six — золото Neutral/Positive, модель
   берёт самый яркий кусок.
5. Короткие и эмодзи-твиты («Mori😻😻😻😻», заголовок Let's Play) дают мало
   сигнала; модель сдвигает их к Positive.

По entity ошибки сконцентрированы в игровых брендах (Rainbow Six, CoD, Apex, RDR,
PUBG) — это основной домен датасета, а не слабое место конкретной игры. Часть
золотых меток самой разметки спорная (позитивный твит про сестру как Neutral,
шутливая «зависимость» как Negative), поэтому потолок качества ниже 100%.
"""


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


def write_error_analysis(df_test, errors, fp, fn, other, observations):
    n_test = len(df_test)
    n_errors = len(errors)
    mean_err_len = errors["text_length"].mean() if n_errors else 0.0
    mean_all_len = df_test["text"].str.len().mean()

    report = (
        "=== АНАЛИЗ ОШИБОК ===\n\n"
        f"split: val={VAL_CSV.name} ({n_test})\n"
        f"Всего ошибок: {n_errors} ({n_errors / n_test * 100:.1f}%)\n"
        f"False Positives (Positive вместо Negative): {len(fp)}\n"
        f"False Negatives (Negative вместо Positive): {len(fn)}\n"
        f"Прочие ошибки: {len(other)}\n\n"
        "=== ПАРЫ ПУТАНИЦЫ (true → pred) ===\n"
        f"{_confusion_pairs(errors)}\n\n"
        "=== ДЛИНА ТЕКСТОВ ===\n"
        f"Средняя длина ошибочных текстов: {mean_err_len:.0f}\n"
        f"Средняя длина всех текстов: {mean_all_len:.0f}\n\n"
        "=== ТОП ENTITY СРЕДИ ОШИБОК ===\n"
        f"{_entity_counts(errors)}\n\n"
        "=== ПРИМЕРЫ FALSE POSITIVES ===\n"
        "предсказала Positive, истинный класс Negative\n"
        f"{_format_examples(fp)}\n\n"
        "=== ПРИМЕРЫ FALSE NEGATIVES ===\n"
        "предсказала Negative, истинный класс Positive\n"
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
    if not FINE_TUNED_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Нет fine-tuned модели: {FINE_TUNED_MODEL_DIR}. "
            "Сначала: python main.py --finetune"
        )

    device = get_device()
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_DIR)
    model.to(device)
    model.eval()

    _, val_df = load_train_val()
    test_texts = val_df["text"].tolist()
    print(f"Val: {len(test_texts)} примеров ({VAL_CSV.name})")

    preds = predict_fine_tuned(test_texts, model, tokenizer, device=device)
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

    fp = errors[
        (errors["pred_label"] == POSITIVE) & (errors["true_label"] == NEGATIVE)
    ]
    fn = errors[
        (errors["pred_label"] == NEGATIVE) & (errors["true_label"] == POSITIVE)
    ]
    other = errors.drop(index=fp.index.union(fn.index))

    print(f"Всего ошибок: {len(errors)}")
    print(f"False Positives: {len(fp)}")
    print(f"False Negatives: {len(fn)}")
    print(f"Прочие: {len(other)}")
    print("\n=== FALSE POSITIVES (сказали good, а было bad) ===")
    for _, row in fp.iterrows():
        print(_console(f"\nТекст: {row['text'][:100]}..."))
        print(f"Истинный класс: {row['true_label']}, Предсказан: {row['pred_label']}")
    print("\n=== FALSE NEGATIVES (сказали bad, а было good) ===")
    for _, row in fn.iterrows():
        print(_console(f"\nТекст: {row['text'][:100]}..."))
        print(f"Истинный класс: {row['true_label']}, Предсказан: {row['pred_label']}")
    print(f"\nСредняя длина ошибочных текстов: {errors['text_length'].mean():.0f}")
    print(f"Средняя длина всех текстов: {df_test['text'].str.len().mean():.0f}")

    if observations is None:
        observations = OBSERVATIONS

    write_error_analysis(df_test, errors, fp, fn, other, observations)
    return df_test, errors, fp, fn, other


if __name__ == "__main__":
    run_error_analysis()
