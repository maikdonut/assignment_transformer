from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

EN_MODEL_NAME = "distilbert-base-uncased"
RU_MODEL_NAME = "distilbert-base-multilingual-cased"

TRAIN_CSV = DATA_DIR / "twitter_training.csv"
VAL_CSV = DATA_DIR / "twitter_validation.csv"

CSV_COLUMNS = ["tweet_id", "entity", "sentiment", "text"]

MAX_LENGTH = 128
BATCH_SIZE = 32

BASELINE_RESULTS_PATH = ROOT / "baseline_results.txt"
