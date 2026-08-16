import json
import re

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from config import (
    BASELINE_MODEL_META_PATH,
    BASELINE_MODEL_PATH,
    BASELINE_RESULTS_PATH,
    CACHE_DIR,
    EN_MODEL_NAME,
    TRAIN_CSV,
    VAL_CSV,
)
from data import load_train_val
from embeddings import cache_fingerprint, get_cls_embeddings, load_encoder


def _pipeline(classifier):
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def make_baseline_pipeline(C=1.0, class_weight=None):
    return _pipeline(
        LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=1000,
            n_jobs=-1,
        )
    )


def make_baseline_candidates():
    return {
        "logreg_default": make_baseline_pipeline(),
        "logreg_balanced": make_baseline_pipeline(class_weight="balanced"),
        "logreg_balanced_C0.1": make_baseline_pipeline(
            C=0.1, class_weight="balanced"
        ),
        "logreg_balanced_C10": make_baseline_pipeline(
            C=10.0, class_weight="balanced"
        ),
        "linear_svc_balanced": _pipeline(
            LinearSVC(class_weight="balanced", max_iter=5000)
        ),
    }


def read_baseline_f1():
    if not BASELINE_RESULTS_PATH.exists():
        return None
    match = re.search(
        r"validation macro F1:\s*([0-9.]+)",
        BASELINE_RESULTS_PATH.read_text(encoding="utf-8"),
    )
    return float(match.group(1)) if match else None


def select_best_baseline(X_train, y_train, cv_splits=3):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    results = []
    best_name = None
    best_score = float("-inf")
    for name, candidate in make_baseline_candidates().items():
        print(f"CV baseline: {name}")
        scores = cross_val_score(
            candidate,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
        )
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        results.append({"name": name, "mean_f1": mean, "std_f1": std})
        print(f"  macro F1: {mean:.4f} ± {std:.4f}")
        if mean > best_score:
            best_name, best_score = name, mean

    model = make_baseline_candidates()[best_name]
    model.fit(X_train, y_train)
    return model, best_name, results


def _save_baseline(model, name, train_fingerprint):
    joblib.dump(model, BASELINE_MODEL_PATH)
    BASELINE_MODEL_META_PATH.write_text(
        json.dumps(
            {
                "candidate": name,
                "train_fingerprint": train_fingerprint,
                "encoder": EN_MODEL_NAME,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Baseline модель сохранена: {BASELINE_MODEL_PATH}")


def _load_embeddings(train_texts, val_texts=None):
    train_cache = CACHE_DIR / "train_cls.npy"
    val_cache = CACHE_DIR / "val_cls.npy"
    tokenizer, encoder, device = load_encoder()
    print(f"Модель: {EN_MODEL_NAME}, device: {device}")
    kwargs = {"tokenizer": tokenizer, "model": encoder, "device": device}
    X_train = get_cls_embeddings(train_texts, cache_path=train_cache, **kwargs)
    X_val = None
    if val_texts is not None:
        X_val = get_cls_embeddings(val_texts, cache_path=val_cache, **kwargs)
    return X_train, X_val


def load_or_fit_baseline():
    train_df, _ = load_train_val()
    train_texts = train_df["text"].tolist()
    fingerprint = cache_fingerprint(train_texts)
    if BASELINE_MODEL_PATH.exists() and BASELINE_MODEL_META_PATH.exists():
        try:
            metadata = json.loads(
                BASELINE_MODEL_META_PATH.read_text(encoding="utf-8")
            )
            if metadata.get("train_fingerprint") == fingerprint:
                print(f"Загружаю baseline: {BASELINE_MODEL_PATH}")
                return joblib.load(BASELINE_MODEL_PATH)
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    print("Baseline отсутствует или устарел; запускаю train-only CV")
    X_train, _ = _load_embeddings(train_texts)
    model, name, _ = select_best_baseline(
        X_train, train_df["sentiment"].to_numpy()
    )
    _save_baseline(model, name, fingerprint)
    return model


def run_baseline():
    train_df, val_df = load_train_val()
    train_texts = train_df["text"].tolist()
    val_texts = val_df["text"].tolist()
    y_train = train_df["sentiment"].to_numpy()
    y_val = val_df["sentiment"].to_numpy()
    print(f"Train: {len(train_texts)} примеров, Val: {len(val_texts)} примеров")

    X_train, X_val = _load_embeddings(train_texts, val_texts)
    print(f"X_train {X_train.shape}, X_val {X_val.shape}")
    model, best_name, cv_results = select_best_baseline(X_train, y_train)
    _save_baseline(model, best_name, cache_fingerprint(train_texts))

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="macro")
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, digits=4)
    cv_text = "\n".join(
        f"  {row['name']}: {row['mean_f1']:.4f} ± {row['std_f1']:.4f}"
        for row in cv_results
    )
    results = (
        f"encoder: {EN_MODEL_NAME}\n"
        f"selection: 3-fold stratified CV on train, metric=macro F1\n"
        f"candidates:\n{cv_text}\n"
        f"selected: {best_name}\n"
        f"split: train={TRAIN_CSV.name} ({len(train_texts)}), "
        f"val={VAL_CSV.name} ({len(val_texts)})\n"
        f"validation macro F1: {f1:.4f}\n"
        f"validation accuracy: {accuracy:.4f}\n\n{report}"
    )
    BASELINE_RESULTS_PATH.write_text(results, encoding="utf-8")
    print(report)
    print(f"Macro F1: {f1:.4f}; Accuracy: {accuracy:.4f}")
    print(f"Результаты сохранены: {BASELINE_RESULTS_PATH}")
    return f1
