# ============================================================
# app.py  —  SentimentAI  (bug-fixed, production-ready)
# ============================================================
# Fixes applied
# ─────────────
#  1. Response + StringIO imported at module top (no duplicates)
#  2. list[dict] → List[dict] for Python 3.8/3.9 compat
#  3. str | None → Optional[str]
#  4. upload_csv: try utf-8 then fall back to latin-1
#  5. get_all_history on dashboard now injects emoji key
#  6. download_csv redirect fixed (was missing return)
# ============================================================

import os
import sys
import json
import sqlite3
import uuid
from datetime import datetime
from io import StringIO
from typing import List, Optional

import joblib
import pandas as pd
from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, jsonify,
    session, Response
)
from werkzeug.utils import secure_filename

# ── Project imports ──────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from utils.preprocessor import clean_text
from utils.visualizer import (
    compute_statistics,
    get_pie_chart_data,
    get_bar_chart_data,
    get_top_words,
    get_recent_predictions,
)

# ════════════════════════════════════════════════════════════
# APP CONFIGURATION
# ════════════════════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = "sentimentai_secret_2024_demo"

UPLOAD_FOLDER   = os.path.join(ROOT, "uploads")
ALLOWED_EXT     = {"csv"}
MAX_CSV_ROWS    = 5_000

MODEL_PATH      = os.path.join(ROOT, "ml", "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(ROOT, "ml", "models", "tfidf_vectorizer.pkl")
DB_PATH         = os.path.join(ROOT, "sentiment_app.db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ════════════════════════════════════════════════════════════
# LOAD ML MODEL (once at startup)
# ════════════════════════════════════════════════════════════
model      = None
vectorizer = None


def load_model():
    global model, vectorizer
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("  Model files not found. Run  python ml/train_model.py  first.")
        return False
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("  Model and vectorizer loaded successfully.")
    return True


# ════════════════════════════════════════════════════════════
# DATABASE
# ════════════════════════════════════════════════════════════
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text    TEXT    NOT NULL,
                cleaned_text  TEXT    NOT NULL,
                sentiment     TEXT    NOT NULL,
                confidence    REAL    NOT NULL,
                source        TEXT    DEFAULT 'single',
                session_id    TEXT,
                created_at    TEXT    DEFAULT (datetime('now','localtime'))
            );

            CREATE TABLE IF NOT EXISTS bulk_sessions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id     TEXT    UNIQUE NOT NULL,
                filename       TEXT    NOT NULL,
                total          INTEGER DEFAULT 0,
                positive_count INTEGER DEFAULT 0,
                negative_count INTEGER DEFAULT 0,
                positive_pct   REAL    DEFAULT 0.0,
                negative_pct   REAL    DEFAULT 0.0,
                created_at     TEXT    DEFAULT (datetime('now','localtime'))
            );
        """)
    print("  Database initialised.")


# ════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ════════════════════════════════════════════════════════════
def _make_emoji(sentiment):
    return "😊" if sentiment == "Positive" else "😢"


def predict_single(text):
    # type: (str) -> dict
    if model is None or vectorizer is None:
        return {"error": "Model not loaded. Run python ml/train_model.py first."}

    cleaned    = clean_text(text)
    # Guard: if cleaning yields empty string, predict on raw lowercased text
    if not cleaned.strip():
        cleaned = text.lower().strip()

    vec        = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    proba      = model.predict_proba(vec)[0]
    confidence = round(float(max(proba)) * 100, 2)

    return {
        "original_text":       text,
        "cleaned_text":        cleaned,
        "sentiment":           prediction,
        "confidence":          confidence,
        "emoji":               _make_emoji(prediction),
        "confidence_bar_class": "bg-success" if prediction == "Positive" else "bg-danger",
    }


def predict_bulk(texts):
    # type: (List[str]) -> List[dict]
    if model is None or vectorizer is None:
        return []

    cleaned_list = [clean_text(t) or t.lower().strip() for t in texts]
    vecs         = vectorizer.transform(cleaned_list)
    predictions  = model.predict(vecs)
    probas       = model.predict_proba(vecs)

    results = []
    for pred, proba, original, cleaned in zip(predictions, probas, texts, cleaned_list):
        confidence = round(float(max(proba)) * 100, 2)
        results.append({
            "original_text": original,
            "text":          original,
            "cleaned_text":  cleaned,
            "sentiment":     pred,
            "confidence":    confidence,
            "emoji":         _make_emoji(pred),
        })
    return results


def _row_to_dict(row):
    """Convert a sqlite3.Row to a plain dict with all required keys."""
    d = dict(row)
    # Normalise column name differences
    d.setdefault("original_text", d.get("input_text", ""))
    d.setdefault("text",          d.get("input_text", ""))
    # Always inject emoji so templates never KeyError
    d["emoji"] = _make_emoji(d.get("sentiment", ""))
    return d


def save_single_to_db(result):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO analysis_history
               (input_text, cleaned_text, sentiment, confidence, source)
               VALUES (?, ?, ?, ?, 'single')""",
            (result["original_text"], result["cleaned_text"],
             result["sentiment"],     result["confidence"])
        )


def save_bulk_to_db(results, filename, session_id):
    stats = compute_statistics(results)
    with get_db() as conn:
        conn.executemany(
            """INSERT INTO analysis_history
               (input_text, cleaned_text, sentiment, confidence, source, session_id)
               VALUES (?, ?, ?, ?, 'bulk', ?)""",
            [(r["original_text"], r["cleaned_text"],
              r["sentiment"],     r["confidence"], session_id)
             for r in results]
        )
        conn.execute(
            """INSERT OR REPLACE INTO bulk_sessions
               (session_id, filename, total, positive_count,
                negative_count, positive_pct, negative_pct)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, filename,
             stats["total"],          stats["positive_count"],
             stats["negative_count"], stats["positive_pct"],
             stats["negative_pct"])
        )


def get_all_history():
    # type: () -> List[dict]
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM analysis_history ORDER BY id DESC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_last_bulk_session_id():
    # type: () -> Optional[str]
    with get_db() as conn:
        row = conn.execute(
            "SELECT session_id FROM bulk_sessions ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row["session_id"] if row else None


def get_bulk_results_by_session(session_id):
    # type: (str) -> List[dict]
    with get_db() as conn:
        rows = conn.execute(
            """SELECT input_text, cleaned_text, sentiment, confidence
               FROM analysis_history
               WHERE session_id = ?
               ORDER BY id""",
            (session_id,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


# ════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    # Show lifetime stats on the index page
    all_history = get_all_history()
    stats       = compute_statistics(all_history) if all_history else None
    return render_template("index.html", result=None, stats=stats)


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()

    if not text:
        flash("Please enter some text to analyse.", "error")
        return redirect(url_for("index"))

    if len(text) > 2000:
        flash("Text is too long. Please keep it under 2,000 characters.", "error")
        return redirect(url_for("index"))

    if model is None:
        flash("Model not loaded. Run  python ml/train_model.py  first.", "error")
        return redirect(url_for("index"))

    result = predict_single(text)

    if "error" in result:
        flash(result["error"], "error")
        return redirect(url_for("index"))

    save_single_to_db(result)

    all_history = get_all_history()
    stats       = compute_statistics(all_history)

    return render_template("index.html", result=result, stats=stats)


@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if "csv_file" not in request.files:
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    file = request.files["csv_file"]

    if not file or file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXT:
        flash("Only CSV files are allowed.", "error")
        return redirect(url_for("index"))

    if model is None:
        flash("Model not loaded. Run  python ml/train_model.py  first.", "error")
        return redirect(url_for("index"))

    # ── Read CSV: try utf-8 first, fall back to latin-1 ─────
    try:
        raw = file.read()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = raw.decode("latin-1")
        df = pd.read_csv(StringIO(content), nrows=MAX_CSV_ROWS)
    except Exception as e:
        flash("Could not parse CSV: {}".format(str(e)), "error")
        return redirect(url_for("index"))

    if df.empty:
        flash("The uploaded CSV appears to be empty.", "error")
        return redirect(url_for("index"))

    # ── Detect text column ───────────────────────────────────
    text_col = None
    lower_cols = {c.lower(): c for c in df.columns}
    for candidate in ["text", "tweet", "comment", "review", "message", "content", "sentence"]:
        if candidate in lower_cols:
            text_col = lower_cols[candidate]
            break

    if text_col is None:
        str_cols = df.select_dtypes(include="object").columns.tolist()
        text_col = str_cols[0] if str_cols else None

    if text_col is None:
        flash("CSV must contain a text/tweet/comment column.", "error")
        return redirect(url_for("index"))

    texts = df[text_col].dropna().astype(str).tolist()

    if not texts:
        flash("No readable text rows found in the CSV.", "error")
        return redirect(url_for("index"))

    # ── Predict + save ───────────────────────────────────────
    results    = predict_bulk(texts)
    session_id = str(uuid.uuid4())
    save_bulk_to_db(results, secure_filename(file.filename), session_id)
    session["last_bulk_session"] = session_id

    flash(
        "Analysed {:,} comments from '{}'.".format(len(results), file.filename),
        "success"
    )
    return redirect(url_for("dashboard"))


@app.route("/dashboard", methods=["GET"])
def dashboard():
    session_id = session.get("last_bulk_session") or get_last_bulk_session_id()

    if session_id:
        results = get_bulk_results_by_session(session_id)
    else:
        results = get_all_history()

    stats    = compute_statistics(results)
    pie_data = get_pie_chart_data(stats)
    bar_data = get_bar_chart_data(results)
    top_pos  = get_top_words(results, "Positive", top_n=15)
    top_neg  = get_top_words(results, "Negative", top_n=15)
    recent   = get_recent_predictions(results, n=10)

    return render_template(
        "dashboard.html",
        stats    = stats,
        pie_json = json.dumps(pie_data),
        bar_json = json.dumps(bar_data),
        top_pos  = top_pos,
        top_neg  = top_neg,
        recent   = recent,
        has_data = stats["total"] > 0,
    )


@app.route("/download_csv", methods=["GET"])
def download_csv():
    session_id = session.get("last_bulk_session") or get_last_bulk_session_id()

    if session_id:
        results = get_bulk_results_by_session(session_id)
    else:
        results = get_all_history()

    if not results:
        flash("No data available to download.", "error")
        return redirect(url_for("dashboard"))

    output = StringIO()
    output.write("text,sentiment,confidence\n")
    for r in results:
        text       = str(r.get("original_text", r.get("input_text", ""))).replace('"', '""')
        sentiment  = r.get("sentiment", "")
        confidence = r.get("confidence", "")
        output.write('"{text}",{sentiment},{confidence}\n'.format(
            text=text, sentiment=sentiment, confidence=confidence
        ))

    output.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
                "attachment; filename=sentiment_results_{}.csv".format(timestamp)
        }
    )


@app.route("/clear_session", methods=["GET"])
def clear_session():
    session.pop("last_bulk_session", None)
    flash("Session cleared. Dashboard now shows all-time history.", "success")
    return redirect(url_for("dashboard"))


@app.route("/api/stats", methods=["GET"])
def api_stats():
    all_rows = get_all_history()
    stats    = compute_statistics(all_rows)
    return jsonify(stats)


@app.route("/api/history", methods=["GET"])
def api_history():
    n    = int(request.args.get("n", 20))
    rows = get_all_history()[:n]
    return jsonify(rows)


# ════════════════════════════════════════════════════════════
# TEMPLATE FILTERS
# ════════════════════════════════════════════════════════════
@app.template_filter("sentiment_badge")
def sentiment_badge(sentiment):
    return "badge-positive" if sentiment == "Positive" else "badge-negative"


@app.template_filter("truncate_text")
def truncate_text(text, length=80):
    text = str(text)
    return text[:length] + "…" if len(text) > length else text


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    init_db()
    model_ok = load_model()
    if not model_ok:
        print("\n  Starting in LIMITED mode — train model first.")
        print("  Run:  python ml/train_model.py\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
