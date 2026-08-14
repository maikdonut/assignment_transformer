import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baseline import load_or_fit_baseline
from config import (
    BATCH_SIZE,
    COMPARISON_RESULTS_PATH,
    FINE_TUNED_MODEL_DIR,
    MAX_LENGTH,
    ROOT,
    VAL_CSV,
)
from data import load_train_val
from embeddings import get_cls_embeddings, get_device, load_encoder

CLASS_NAMES = ["Irrelevant", "Negative", "Neutral", "Positive"]

EXAMPLE_TEXTS = [
    "This movie was absolutely fantastic!",
    "Terrible, waste of my time.",
    "It was okay, nothing special.",
    "Best film I've seen this year!",
    "Boring and too long.",
    "Can't wait to play Borderlands tonight!",
    "This stream is so boring I might just leave.",
    "The new patch notes dropped, nothing crazy.",
    "Why is this tweet on my gaming feed about taxes?",
]

CM_FINETUNED_PATH = ROOT / "confusion_matrix_finetuned.png"
CM_BASELINE_PATH = ROOT / "confusion_matrix_baseline.png"


def _as_list(texts):
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def _id2label(model):
    return {int(k): v for k, v in model.config.id2label.items()}


def predict_fine_tuned(texts, model, tokenizer, device=None, batch_size=BATCH_SIZE):
    texts = _as_list(texts)
    if device is None:
        device = next(model.parameters()).device

    id2label = _id2label(model)
    predictions = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)
        pred_ids = torch.argmax(probs, dim=1)

        for i, text in enumerate(batch_texts):
            label_id = int(pred_ids[i].item())
            predictions.append(
                {
                    "text": text,
                    "prediction": id2label[label_id],
                    "label_id": label_id,
                    "probabilities": probs[i].cpu().numpy(),
                }
            )

    return predictions


def predict_baseline(texts, clf, tokenizer, encoder, device=None):
    texts = _as_list(texts)
    embeddings = get_cls_embeddings(
        texts,
        tokenizer=tokenizer,
        model=encoder,
        device=device,
    )
    pred_labels = clf.predict(embeddings)
    probs = clf.predict_proba(embeddings) if hasattr(clf, "predict_proba") else None

    results = []
    for i, text in enumerate(texts):
        results.append(
            {
                "text": text,
                "prediction": str(pred_labels[i]),
                "probabilities": None if probs is None else probs[i],
            }
        )
    return results


def save_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Сохранено: {path}")
    return cm


def print_examples(preds_ft, preds_base, class_names, baseline_classes):
    print("\n=== Примеры ===")
    for ft, base in zip(preds_ft, preds_base):
        print(f"\nТекст: {ft['text']}")
        ft_probs = ", ".join(
            f"{class_names[i]}={ft['probabilities'][i]:.2f}"
            for i in range(len(class_names))
        )
        base_probs = ", ".join(
            f"{label}={p:.2f}" for label, p in zip(baseline_classes, base["probabilities"])
        )
        print(f"Fine-tuned: {ft['prediction']} ({ft_probs})")
        print(f"Baseline:   {base['prediction']} ({base_probs})")
        print(f"Совпадают: {ft['prediction'] == base['prediction']}")


def run_compare():
    if not FINE_TUNED_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Нет fine-tuned модели: {FINE_TUNED_MODEL_DIR}. Сначала: python main.py --finetune"
        )

    device = get_device()
    print(f"device: {device}")

    tokenizer_ft = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR)
    model_ft = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_DIR)
    model_ft.to(device)
    model_ft.eval()

    clf = load_or_fit_baseline()
    encoder_tokenizer, encoder, encoder_device = load_encoder(device=device)

    _, val_df = load_train_val()
    test_texts = val_df["text"].tolist()
    test_labels = val_df["sentiment"].astype(str).tolist()
    print(f"Val: {len(test_texts)} примеров ({VAL_CSV.name})")

    preds_ft_examples = predict_fine_tuned(
        EXAMPLE_TEXTS, model_ft, tokenizer_ft, device=device
    )
    preds_base_examples = predict_baseline(
        EXAMPLE_TEXTS, clf, encoder_tokenizer, encoder, device=encoder_device
    )
    print_examples(preds_ft_examples, preds_base_examples, CLASS_NAMES, clf.classes_)

    print("\n=== Предсказания на validation ===")
    preds_ft_all = predict_fine_tuned(
        test_texts, model_ft, tokenizer_ft, device=device
    )
    preds_base_all = predict_baseline(
        test_texts, clf, encoder_tokenizer, encoder, device=encoder_device
    )
    y_pred_ft = [p["prediction"] for p in preds_ft_all]
    y_pred_base = [p["prediction"] for p in preds_base_all]

    report_ft = classification_report(test_labels, y_pred_ft, labels=CLASS_NAMES)
    report_base = classification_report(test_labels, y_pred_base, labels=CLASS_NAMES)
    f1_ft = f1_score(test_labels, y_pred_ft, average="macro", labels=CLASS_NAMES)
    acc_ft = accuracy_score(test_labels, y_pred_ft)
    f1_base = f1_score(test_labels, y_pred_base, average="macro", labels=CLASS_NAMES)
    acc_base = accuracy_score(test_labels, y_pred_base)
    improvement = (f1_ft - f1_base) / f1_base * 100

    print("=== Fine-tuned Model ===")
    print(report_ft)
    print("=== Baseline Model ===")
    print(report_base)
    print("=== Сравнение ===")
    print(f"Fine-tuned F1: {f1_ft:.4f}, Accuracy: {acc_ft:.4f}")
    print(f"Baseline F1: {f1_base:.4f}, Accuracy: {acc_base:.4f}")
    print(f"Улучшение F1: {improvement:.2f}%")

    save_confusion_matrix(
        test_labels,
        y_pred_ft,
        "Confusion Matrix - Fine-tuned Model",
        CM_FINETUNED_PATH,
    )
    save_confusion_matrix(
        test_labels,
        y_pred_base,
        "Confusion Matrix - Baseline Model",
        CM_BASELINE_PATH,
    )

    results = (
        "=== Сравнение моделей ===\n\n"
        f"split: val={VAL_CSV.name} ({len(test_texts)})\n\n"
        "Fine-tuned Model:\n"
        f"  F1 (macro): {f1_ft:.4f}\n"
        f"  Accuracy: {acc_ft:.4f}\n\n"
        "Baseline Model (CLS + Logistic Regression):\n"
        f"  F1 (macro): {f1_base:.4f}\n"
        f"  Accuracy: {acc_base:.4f}\n\n"
        f"Улучшение F1: {improvement:.2f}%\n\n"
        "=== Fine-tuned classification report ===\n"
        f"{report_ft}\n"
        "=== Baseline classification report ===\n"
        f"{report_base}"
    )
    COMPARISON_RESULTS_PATH.write_text(results, encoding="utf-8")
    print(f"Результаты сохранены: {COMPARISON_RESULTS_PATH}")
    return f1_ft, f1_base
