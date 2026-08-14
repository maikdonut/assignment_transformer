import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification

from config import (
    EN_MODEL_NAME,
    FINE_TUNED_MODEL_DIR,
    FINE_TUNED_RESULTS_PATH,
    FINETUNE_BATCH_SIZE,
    LEARNING_RATE,
    MAX_LENGTH,
    NUM_EPOCHS,
    TRAIN_CSV,
    VAL_CSV,
)
from data import load_train_val
from embeddings import get_device
from tokenization import load_tokenizer

BASELINE_MACRO_F1 = 0.6017


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def encode_labels(series, label2id):
    return [label2id[label] for label in series.tolist()]


def train_epoch(model, dataloader, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0

    progress = tqdm(dataloader, desc=f"Train {epoch}/{num_epochs}", leave=True)
    for batch in progress:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val", leave=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro")

    return accuracy, f1, true_labels, predictions


def run_finetune(max_samples=None):
    train_df, val_df = load_train_val()
    labels_sorted = sorted(train_df["sentiment"].unique())
    label2id = {label: i for i, label in enumerate(labels_sorted)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(label2id)

    if max_samples is not None:
        n_train = min(max_samples, len(train_df))
        train_df = train_df.sample(n=n_train, random_state=42).reset_index(drop=True)
        print(f"Ограничение train: {n_train} примеров (--max-samples)")

    train_texts = train_df["text"].tolist()
    val_texts = val_df["text"].tolist()
    train_labels = encode_labels(train_df["sentiment"], label2id)
    val_labels = encode_labels(val_df["sentiment"], label2id)

    tokenizer = load_tokenizer()
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=FINETUNE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=FINETUNE_BATCH_SIZE,
        num_workers=0,
    )

    device = get_device()
    model = AutoModelForSequenceClassification.from_pretrained(
        EN_MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Модель: {EN_MODEL_NAME}, device: {device}")
    print(f"Классы ({num_labels}): {labels_sorted}")
    print(f"Train: {len(train_texts)} примеров, Val: {len(val_texts)} примеров")
    print(f"Эпохи: {NUM_EPOCHS}, batch: {FINETUNE_BATCH_SIZE}, lr: {LEARNING_RATE}")

    val_acc = 0.0
    val_f1 = 0.0
    true_labels = []
    predictions = []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, epoch + 1, NUM_EPOCHS
        )
        val_acc, val_f1, true_labels, predictions = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        print(f"Val F1: {val_f1:.4f}")
        print("-" * 50)

    target_names = [id2label[i] for i in range(num_labels)]
    report = classification_report(
        true_labels,
        predictions,
        target_names=target_names,
        digits=2,
    )
    print(report)
    print(f"Macro F1: {val_f1:.4f} (baseline: {BASELINE_MACRO_F1:.4f})")

    FINE_TUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINE_TUNED_MODEL_DIR)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_DIR)
    print(f"Модель сохранена: {FINE_TUNED_MODEL_DIR}")

    results = (
        f"model: {EN_MODEL_NAME}\n"
        f"task: sequence classification, num_labels={num_labels}\n"
        f"epochs: {NUM_EPOCHS}\n"
        f"batch_size: {FINETUNE_BATCH_SIZE}\n"
        f"learning_rate: {LEARNING_RATE}\n"
        f"split: train={TRAIN_CSV.name} ({len(train_texts)}), "
        f"val={VAL_CSV.name} ({len(val_texts)})\n"
        f"Final Validation Accuracy: {val_acc:.4f}\n"
        f"Final Validation F1: {val_f1:.4f}\n"
        f"baseline macro F1: {BASELINE_MACRO_F1:.4f}\n\n"
        f"{report}"
    )
    FINE_TUNED_RESULTS_PATH.write_text(results, encoding="utf-8")
    print(f"Результаты сохранены: {FINE_TUNED_RESULTS_PATH}")
    return val_acc, val_f1
