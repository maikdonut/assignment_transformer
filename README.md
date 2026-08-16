# Sentiment Analysis с DistilBERT

## Описание

Проект по анализу тональности твитов (Twitter Entity Sentiment) с трансформерами. Четыре класса: **Irrelevant**, **Negative**, **Neutral**, **Positive**.

Пайплайн: EDA и preprocessing → токенизация → CLS-эмбеддинги DistilBERT → подбор frozen baseline → fine-tuning DistilBERT → сравнение моделей → динамический анализ ошибок → Gradio-демо.

## Структура

Логическая группировка (файлы лежат в корне, без лишних пакетов):

**Данные**
- `data/twitter_training.csv`, `data/twitter_validation.csv` — датасет
- `preprocessing.py`, `data.py` — единая очистка для train и inference
- `eda.py`, `eda_report.txt`, `eda_plots/` — анализ датасета и выводы

**Обучение**
- `tokenization.py`, `embeddings.py` — токенизация и CLS-эмбеддинги
- `baseline.py` — train-only CV для Logistic Regression / LinearSVC на CLS
- `finetune.py` — дообучение DistilBERT
- `fine_tuned_model/` — локальный checkpoint (веса в `.gitignore`)

**Сравнение и анализ**
- `inference.py` — единая загрузка локальной/Hub-модели и предсказание
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

Веса `fine_tuned_model/` в git не хранятся. Если локальной папки нет,
модель автоматически загружается из `martinjob/twitter-sentiment-distilbert`
на Hugging Face Hub. Для полностью offline-запуска заранее поместите checkpoint
в `fine_tuned_model/`.

Повторное обучение требуется только для воспроизведения эксперимента:

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

### Baseline модель (лучший frozen CLS classifier)

- F1 (macro): 0.6119
- Accuracy: 0.6260

Улучшение F1: 56.98%

`error_analysis.txt` при каждом запуске строится из актуальных предсказаний:
ошибки, пары путаницы, FP/FN, длины, URL, хештеги и top entities.

## Команды

```bash
uv run python main.py              # токенизация, эмбеддинги, baseline
uv run python main.py --eda        # EDA, отчёт и графики
uv run python main.py --attention  # визуализация attention
uv run python main.py --finetune   # дообучение DistilBERT
uv run python main.py --compare    # сравнение с baseline
uv run python main.py --errors     # анализ ошибок
uv run python app.py               # Gradio-демо
uv run pytest -q                   # быстрые тесты (после uv sync --extra dev)
```

## Использование в коде

```python
from inference import predict_sentiments

result = predict_sentiments("Your text here")[0]
print(result["prediction"], result["probabilities"])
```

## Воспроизводимость

- Preprocessing сохраняет пунктуацию, URL и эмодзи, но нормализует пробелы и
  удаляет пустые/невалидные строки.
- CLS-кэш имеет fingerprint текстов, encoder и `max_length`; устаревший кэш
  автоматически пересчитывается.
- Варианты baseline выбираются по stratified CV только на train. Validation
  используется один раз для итогового отчёта.
- Для публикации локального checkpoint: `uv run hf auth login`, затем
  `uv run hf upload martinjob/twitter-sentiment-distilbert fine_tuned_model .`.

## Требования

- Python 3.10+
- transformers, torch, pandas, scikit-learn
- matplotlib, seaborn
- gradio (для демо)
