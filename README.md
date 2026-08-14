# Sentiment Analysis с DistilBERT

## Описание

Проект по анализу тональности твитов (Twitter Entity Sentiment) с трансформерами. Четыре класса: **Irrelevant**, **Negative**, **Neutral**, **Positive**.

Пайплайн: токенизация → CLS-эмбеддинги DistilBERT → baseline (логистическая регрессия) → fine-tuning DistilBERT → сравнение моделей → анализ ошибок → Gradio-демо.

## Структура

Логическая группировка (файлы лежат в корне, без лишних пакетов):

**Данные**
- `data/twitter_training.csv`, `data/twitter_validation.csv` — датасет
- `data.py`, `config.py` — загрузка и пути

**Обучение**
- `tokenization.py`, `embeddings.py` — токенизация и CLS-эмбеддинги
- `baseline.py` — Logistic Regression на CLS
- `finetune.py` — дообучение DistilBERT
- `fine_tuned_model/` — сохранённая модель (веса в `.gitignore`)

**Сравнение и анализ**
- `compare.py` — инференс и сравнение с baseline
- `error_analysis.py` — False Positive / False Negative и паттерны ошибок
- `comparison_results.txt`, `error_analysis.txt`, `*_results.txt`

**Демо**
- `app.py` — Gradio-интерфейс
- `main.py` — единая точка входа по дням (`--finetune`, `--compare`, `--errors`)

## Установка

Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Веса `fine_tuned_model/` в git не хранятся. Если папки нет, сначала дообучите модель:

```bash
uv run python main.py --finetune
```

## Запуск демо

```bash
uv run python app.py
```

Откройте http://127.0.0.1:7860

## Результаты

Validation: `twitter_validation.csv` (1000 примеров).

### Fine-tuned модель

- F1 (macro): 0.9606
- Accuracy: 0.9620

### Baseline модель (CLS + Logistic Regression)

- F1 (macro): 0.6015
- Accuracy: 0.6210

Улучшение F1: 59.71%

На 1000 val-примерах у fine-tuned модели 38 ошибок (3.8%). Почти нет смены полярности Positive ↔ Negative (1 FP, 0 FN); чаще путаются Neutral / Irrelevant / Positive. Подробности — в `error_analysis.txt`.

## Команды

```bash
uv run python main.py              # токенизация, эмбеддинги, baseline
uv run python main.py --attention  # визуализация attention
uv run python main.py --finetune   # дообучение DistilBERT
uv run python main.py --compare    # сравнение с baseline
uv run python main.py --errors     # анализ ошибок
uv run python app.py               # Gradio-демо
```

## Использование в коде

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")

inputs = tokenizer("Your text here", return_tensors="pt", truncation=True, max_length=128)
outputs = model(**inputs)
pred_id = int(torch.argmax(outputs.logits, dim=1))
label = model.config.id2label[pred_id]
```

## Требования

- Python 3.10+
- transformers, torch, pandas, scikit-learn
- matplotlib, seaborn
- gradio (для демо)
