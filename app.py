import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import FINE_TUNED_MODEL_DIR, MAX_LENGTH
from embeddings import get_device

_tokenizer = None
_model = None
_device = None
_id2label = None


def load_sentiment_model():
    global _tokenizer, _model, _device, _id2label
    if _model is not None:
        return

    if not FINE_TUNED_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Нет fine-tuned модели: {FINE_TUNED_MODEL_DIR}. "
            "Сначала: python main.py --finetune"
        )

    _device = get_device()
    _tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_DIR)
    _model.to(_device)
    _model.eval()
    _id2label = {int(k): v for k, v in _model.config.id2label.items()}
    print(f"Модель загружена: {FINE_TUNED_MODEL_DIR} (device={_device})")


def predict_sentiment(text):
    if text is None or not str(text).strip():
        return "Введите текст для анализа."

    load_sentiment_model()
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(_device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0]
    pred = int(torch.argmax(probs).item())
    label_map = _id2label

    lines = [f"Prediction: {label_map.get(pred, pred)}", "", "Probabilities:"]
    for i, prob in enumerate(probs):
        lines.append(f"{label_map.get(i, i)}: {prob.item() * 100:.2f}%")
    return "\n".join(lines)


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Введите текст для анализа..."),
    outputs=gr.Textbox(label="Результат"),
    title="Sentiment Analysis с DistilBERT",
    description=(
        "Введите текст — модель определит тональность "
        "(Irrelevant / Negative / Neutral / Positive) на датасете Twitter Entity Sentiment."
    ),
    examples=[
        ["This movie was absolutely fantastic!"],
        ["Terrible, waste of my time."],
        ["It was okay, nothing special."],
        ["Can't wait to play Borderlands tonight!"],
        ["Why is this tweet on my gaming feed about taxes?"],
    ],
)


if __name__ == "__main__":
    load_sentiment_model()
    print("Откройте http://127.0.0.1:7860", flush=True)
    demo.launch(server_name="127.0.0.1", server_port=7860)
