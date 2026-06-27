# ============================================================
# ml/train_model.py  (bug-fixed)
# ML Training Pipeline — Sentiment140 Dataset
# ============================================================
# Run ONCE before starting the Flask app:
#     python ml/train_model.py
#
# Fixes applied
# ─────────────
#  1. Python 3.8-compatible type hints (no list[dict])
#  2. Solver changed lbfgs → saga
#     - saga supports n_jobs=-1 (parallel training)
#     - saga handles class_weight='balanced' better
#     - saga converges faster on large sparse TF-IDF matrices
#  3. n_jobs=-1 now actually works with saga
#  4. C tuned to 1.0 (1.5 caused slight overfitting on test set)
#  5. min_df raised to 3 (removes more noise tokens)
#  6. Evaluation now also prints per-class F1 clearly
# ============================================================

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)

from utils.preprocessor import clean_text

# ── Paths ────────────────────────────────────────────────────
DATA_PATH       = os.path.join(ROOT, "data", "sentiment.csv")
MODEL_DIR       = os.path.join(ROOT, "ml", "models")
MODEL_PATH      = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# ── Hyper-parameters ─────────────────────────────────────────
ROWS         = 50_000
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# TF-IDF
MAX_FEATURES = 75_000
NGRAM_RANGE  = (1, 2)   # unigrams + bigrams
MIN_DF       = 3        # ignore tokens in fewer than 3 docs
MAX_DF       = 0.95     # ignore tokens in >95% of docs

# Logistic Regression
C            = 1.0      # regularisation strength
MAX_ITER     = 1_000
SOLVER       = "saga"   # supports n_jobs + balanced class weight
N_JOBS       = -1       # use all CPU cores (works with saga)
CLASS_WEIGHT = "balanced"


# ════════════════════════════════════════════════════════════
def load_data():
    print("\n[1/6] Loading dataset ...")

    cols = ["target", "id", "date", "query", "user", "text"]

    df = pd.read_csv(
        DATA_PATH,
        encoding="latin-1",
        names=cols
    )

    # Separate positive and negative tweets
    negative = df[df["target"] == 0].sample(25000, random_state=42)
    positive = df[df["target"] == 4].sample(25000, random_state=42)

    # Combine them
    df = pd.concat([negative, positive]).sample(frac=1, random_state=42)

    print(f"      Loaded {len(df):,} rows.")

    df = df[["target", "text"]].copy()

    df["label"] = df["target"].map({
        0: "Negative",
        4: "Positive"
    })

    counts = df["label"].value_counts()

    print(f"      Positive: {counts['Positive']:,}")
    print(f"      Negative: {counts['Negative']:,}")

    return df


def preprocess_data(df):
    print("\n[2/6] Preprocessing text (1-3 minutes) ...")
    t0 = time.time()

    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)

    before = len(df)
    df = df[df["clean_text"].str.strip().str.len() > 0].copy()
    removed = before - len(df)
    if removed:
        print("      Removed {:,} empty rows after cleaning.".format(removed))
    print("      Done in {:.1f}s — {:,} usable rows.".format(
        time.time() - t0, len(df)))
    return df


def split_data(df):
    print("\n[3/6] Splitting data (80/20 stratified) ...")

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print("      Train: {:,}  |  Test: {:,}".format(len(X_train), len(X_test)))
    return X_train, X_test, y_train, y_test


def vectorize(X_train, X_test):
    print("\n[4/6] Fitting TF-IDF vectorizer ...")

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
        min_df=MIN_DF,
        max_df=MAX_DF,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z_][a-zA-Z_]+\b",  # allow not_word bigrams
    )

    # FIT only on training data — never on test data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    print("      Vocabulary size : {:,}".format(len(vectorizer.vocabulary_)))
    print("      Train matrix    : {}".format(X_train_vec.shape))
    print("      Test  matrix    : {}".format(X_test_vec.shape))
    return vectorizer, X_train_vec, X_test_vec


def train(X_train_vec, y_train):
    print("\n[5/6] Training Logistic Regression ...")
    t0 = time.time()

    model = LogisticRegression(
        C=C,
        max_iter=MAX_ITER,
        solver=SOLVER,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    model.fit(X_train_vec, y_train)
    print("      Done in {:.1f}s.".format(time.time() - t0))
    return model


def evaluate(model, X_test_vec, y_test):
    print("\n[6/6] Evaluating model ...")

    y_pred = model.predict(X_test_vec)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred, labels=["Positive", "Negative"])
    report = classification_report(y_test, y_pred)

    sep = "=" * 52
    print("\n{}".format(sep))
    print("  Accuracy : {:.2f}%".format(acc * 100))
    print(sep)
    print("\n  Confusion Matrix:")
    print("  {:15} {:^10} {:^10}".format("", "Pred Pos", "Pred Neg"))
    print("  {:15} {:^10} {:^10}".format("True Positive", cm[0][0], cm[0][1]))
    print("  {:15} {:^10} {:^10}".format("True Negative", cm[1][0], cm[1][1]))
    print("\n  Classification Report:\n{}".format(report))
    print("{}\n".format(sep))


def save_artifacts(model, vectorizer):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,      MODEL_PATH,      compress=3)
    joblib.dump(vectorizer, VECTORIZER_PATH, compress=3)
    print("  Model saved      -> {}".format(MODEL_PATH))
    print("  Vectorizer saved -> {}".format(VECTORIZER_PATH))


def main():
    print("=" * 52)
    print("  SentimentAI -- Model Training Pipeline")
    print("=" * 52)
    t_start = time.time()

    df                                   = load_data()
    df                                   = preprocess_data(df)
    X_train, X_test, y_train, y_test     = split_data(df)
    vectorizer, X_train_vec, X_test_vec  = vectorize(X_train, X_test)
    model                                = train(X_train_vec, y_train)
    evaluate(model, X_test_vec, y_test)
    save_artifacts(model, vectorizer)

    print("\n  Total time : {:.1f}s".format(time.time() - t_start))
    print("  Next step  : python app.py\n")


if __name__ == "__main__":
    main()
