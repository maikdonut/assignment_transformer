from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    BASELINE_RESULTS_PATH,
    CACHE_DIR,
    EN_MODEL_NAME,
    TRAIN_CSV,
    VAL_CSV,
)
from data import load_train_val
from embeddings import get_cls_embeddings, load_encoder


def run_baseline():
    train_df, val_df = load_train_val()
    train_texts = train_df["text"].tolist()
    y_train = train_df["sentiment"].to_numpy()
    val_texts = val_df["text"].tolist()
    y_test = val_df["sentiment"].to_numpy()

    print(f"Train: {len(train_texts)} примеров, Val: {len(val_texts)} примеров")
    print(f"Классы: {sorted(set(y_train))}")

    train_cache = CACHE_DIR / "train_cls.npy"
    val_cache = CACHE_DIR / "val_cls.npy"
    encoder_kwargs = {}
    if train_cache.exists() and val_cache.exists():
        print("Кэш эмбеддингов найден, DistilBERT не загружаю")
    else:
        tokenizer, model, device = load_encoder()
        print(f"Модель: {EN_MODEL_NAME}, device: {device}")
        encoder_kwargs = dict(tokenizer=tokenizer, model=model, device=device)

    X_train = get_cls_embeddings(
        train_texts,
        cache_path=train_cache,
        **encoder_kwargs,
    )
    X_test = get_cls_embeddings(
        val_texts,
        cache_path=val_cache,
        **encoder_kwargs,
    )
    print(f"X_train {X_train.shape}, X_test {X_test.shape}")

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, n_jobs=-1)),
        ]
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(report)
    print(f"Macro F1: {f1:.4f}")

    results = (
        f"model: {EN_MODEL_NAME}\n"
        f"classifier: StandardScaler + LogisticRegression(max_iter=1000, n_jobs=-1)\n"
        f"split: train={TRAIN_CSV.name} ({len(train_texts)}), "
        f"val={VAL_CSV.name} ({len(val_texts)})\n"
        f"macro F1: {f1:.4f}\n\n"
        f"{report}"
    )
    BASELINE_RESULTS_PATH.write_text(results, encoding="utf-8")
    print(f"Результаты сохранены: {BASELINE_RESULTS_PATH}")
    return f1
