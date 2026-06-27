# ============================================================
# config.py — SentimentAI
# Central configuration. Import anywhere with:
#     from config import Config
# ============================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # Flask
    SECRET_KEY    = os.environ.get("SECRET_KEY", "sentimentai_secret_2024_demo")
    DEBUG         = os.environ.get("DEBUG", "True") == "True"
    HOST          = os.environ.get("HOST", "0.0.0.0")
    PORT          = int(os.environ.get("PORT", 5000))

    # Paths
    DATA_PATH       = os.path.join(BASE_DIR, "data",       "sentiment.csv")
    MODEL_PATH      = os.path.join(BASE_DIR, "ml", "models", "sentiment_model.pkl")
    VECTORIZER_PATH = os.path.join(BASE_DIR, "ml", "models", "tfidf_vectorizer.pkl")
    DB_PATH         = os.path.join(BASE_DIR, "sentiment_app.db")
    UPLOAD_FOLDER   = os.path.join(BASE_DIR, "uploads")

    # Upload limits
    ALLOWED_EXTENSIONS = {"csv"}
    MAX_CSV_ROWS        = 5_000

    # Model training
    ROWS         = 50_000
    TEST_SIZE    = 0.20
    RANDOM_STATE = 42
    MAX_FEATURES = 75_000
    NGRAM_RANGE  = (1, 2)
    MIN_DF       = 3
    MAX_DF       = 0.95
    C            = 1.0
    MAX_ITER     = 1_000
    SOLVER       = "saga"
    CLASS_WEIGHT = "balanced"
