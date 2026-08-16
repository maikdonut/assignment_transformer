import gradio as gr

from inference import get_sentiment_model, predict_sentiments


def load_sentiment_model():
    return get_sentiment_model()


def predict_sentiment(text):
    if text is None or not str(text).strip():
        return "Введите текст для анализа."

    model = load_sentiment_model()
    result = predict_sentiments(text, model)[0]
    lines = [f"Prediction: {result['prediction']}", "", "Probabilities:"]
    for label_id, probability in enumerate(result["probabilities"]):
        label = model.id2label.get(label_id, label_id)
        lines.append(f"{label}: {probability * 100:.2f}%")
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
