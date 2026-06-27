# ============================================================
# utils/visualizer.py  (bug-fixed)
# Chart Data & Word Analytics Generator
# ============================================================
# Fixes applied
# ─────────────
#  1. list[dict] → Python 3.8-compatible type comments
#  2. get_top_words: robust key lookup chain
#     (original_text → text → input_text) so it works for
#     both single-prediction dicts and DB row dicts
#  3. compute_statistics: guards against division-by-zero
#     more explicitly and returns floats not ints for pcts
#  4. get_bar_chart_data: confidence stored as 0–100 float,
#     bucket index clamped correctly
# ============================================================

import re
from collections import Counter

import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

_STOPWORDS = set(stopwords.words("english"))

_EXTRA_NOISE = {
    "im", "ive", "id", "dont", "cant", "wont", "didnt", "isnt",
    "wasnt", "arent", "wouldnt", "shouldnt", "couldnt", "its",
    "thats", "theyre", "youre", "were", "weve", "theyd", "youd",
    "get", "got", "go", "going", "know", "think", "want", "make",
    "like", "just", "one", "would", "could", "also", "even",
    "still", "really", "amp", "rt", "via", "http", "https",
    "www", "com", "co", "ly", "bit", "thi", "ha", "wa", "say",
    "said", "will", "us", "good", "day", "time", "need",
}

_ALL_NOISE = _STOPWORDS | _EXTRA_NOISE


# ── Helpers ──────────────────────────────────────────────────

def _get_text(record):
    # type: (dict) -> str
    """Safely extract raw text from any result/row dict."""
    return str(
        record.get("original_text")
        or record.get("text")
        or record.get("input_text")
        or ""
    )


def _empty_stats():
    # type: () -> dict
    return {
        "total":          0,
        "positive_count": 0,
        "negative_count": 0,
        "positive_pct":   0.0,
        "negative_pct":   0.0,
        "avg_confidence": 0.0,
    }


# ── Public API ───────────────────────────────────────────────

def compute_statistics(results):
    # type: (list) -> dict
    """
    Compute aggregate sentiment statistics from a list of
    prediction result dicts.
    """
    total = len(results)
    if total == 0:
        return _empty_stats()

    positive_count = sum(1 for r in results if r.get("sentiment") == "Positive")
    negative_count = total - positive_count

    positive_pct = round((positive_count / total) * 100, 2)
    negative_pct = round((negative_count / total) * 100, 2)

    confidences = [r.get("confidence", 0) for r in results]
    avg_confidence = round(sum(confidences) / total, 2) if confidences else 0.0

    return {
        "total":           total,
        "positive_count":  positive_count,
        "negative_count":  negative_count,
        "positive_pct":    positive_pct,
        "negative_pct":    negative_pct,
        "avg_confidence":  avg_confidence,
    }


def get_pie_chart_data(stats):
    # type: (dict) -> dict
    """Chart.js-ready data for the sentiment Doughnut chart."""
    return {
        "labels": ["Positive", "Negative"],
        "datasets": [{
            "data":                 [stats["positive_count"], stats["negative_count"]],
            "backgroundColor":      ["#22c55e", "#ef4444"],
            "hoverBackgroundColor": ["#16a34a", "#dc2626"],
            "borderColor":          ["#1e293b", "#1e293b"],
            "borderWidth":          3,
        }]
    }


def get_bar_chart_data(results):
    # type: (list) -> dict
    """
    Chart.js-ready data for confidence distribution bar chart.
    10 buckets: 0-10%, 10-20%, …, 90-100%.
    Confidence values are stored as 0.0–100.0 floats.
    """
    buckets    = ["{}-{}%".format(i * 10, (i + 1) * 10) for i in range(10)]
    pos_counts = [0] * 10
    neg_counts = [0] * 10

    for r in results:
        conf = float(r.get("confidence", 0))       # already 0–100
        idx  = min(int(conf // 10), 9)             # clamp to 0–9
        if r.get("sentiment") == "Positive":
            pos_counts[idx] += 1
        else:
            neg_counts[idx] += 1

    return {
        "labels": buckets,
        "datasets": [
            {
                "label":           "Positive",
                "data":            pos_counts,
                "backgroundColor": "rgba(34,197,94,0.75)",
                "borderColor":     "rgba(34,197,94,1)",
                "borderWidth":     2,
                "borderRadius":    6,
            },
            {
                "label":           "Negative",
                "data":            neg_counts,
                "backgroundColor": "rgba(239,68,68,0.75)",
                "borderColor":     "rgba(239,68,68,1)",
                "borderWidth":     2,
                "borderRadius":    6,
            },
        ]
    }


def get_top_words(results, sentiment, top_n=15):
    # type: (list, str, int) -> list
    """
    Extract most frequent meaningful words from comments that
    match the given sentiment label.

    Returns list of {"word": str, "count": int} dicts.
    """
    corpus = " ".join(
        _get_text(r)
        for r in results
        if r.get("sentiment") == sentiment
    )

    corpus = corpus.lower()
    corpus = re.sub(r"https?://\S+|www\.\S+", " ", corpus)
    corpus = re.sub(r"@\w+", " ", corpus)
    corpus = re.sub(r"[^a-z\s]", " ", corpus)
    corpus = re.sub(r"\s+", " ", corpus).strip()

    tokens = [
        w for w in corpus.split()
        if w not in _ALL_NOISE and len(w) >= 3
    ]

    counter = Counter(tokens)
    return [
        {"word": word, "count": count}
        for word, count in counter.most_common(top_n)
    ]


def get_recent_predictions(results, n=10):
    # type: (list, int) -> list
    """Return the last N prediction records, newest first."""
    return list(reversed(results[-n:]))
