# ============================================================
# utils/preprocessor.py  (bug-fixed)
# Text Cleaning & NLP Preprocessing Pipeline
# ============================================================
# Fixes applied
# ─────────────
#  1. Python 3.8-compatible type hints (no list[dict])
#  2. Sentiment-bearing ADVERBS preserved (absolutely, very…)
#  3. Negation bigrams injected BEFORE stemming:
#       "not happy" → tokens include "not_happy"
#     This prevents the model seeing "not" and "happy" as
#     independent positive/negative signals.
#  4. Repeated characters collapsed: "loooove" → "love"
#  5. Common contractions expanded before cleaning
# ============================================================

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Auto-download required NLTK assets ──────────────────────
for _pkg in ("stopwords", "punkt"):
    try:
        nltk.data.find(
            "tokenizers/{}".format(_pkg)
            if _pkg.startswith("punkt")
            else "corpora/{}".format(_pkg)
        )
    except LookupError:
        nltk.download(_pkg, quiet=True)

# ── Singletons ───────────────────────────────────────────────
_stemmer   = PorterStemmer()
_stopwords = set(stopwords.words("english"))

# Words with direct sentiment value — NEVER remove
_SENTIMENT_PRESERVE = {
    # negations
    "not", "no", "nor", "never", "neither", "without",
    # sentiment adjectives
    "good", "bad", "best", "worst", "great", "terrible",
    "awful", "amazing", "horrible", "excellent", "poor",
    "happy", "sad", "angry", "excited", "disappointed",
    "love", "hate", "like", "dislike", "enjoy", "loathe",
    "fantastic", "dreadful", "wonderful", "pathetic",
    "superb", "disgusting", "brilliant", "boring",
    "perfect", "broken", "useless", "helpful",
    # sentiment adverbs (FIX: these were being stripped before)
    "absolutely", "totally", "completely", "truly", "really",
    "very", "highly", "deeply", "strongly", "incredibly",
    "extremely", "utterly", "barely", "hardly", "scarcely",
}

_FILTERED_STOPS = _stopwords - _SENTIMENT_PRESERVE

# ── Contraction expansion map ────────────────────────────────
_CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "i'm": "i am", "i've": "i have", "i'll": "i will",
    "i'd": "i would", "you're": "you are", "you've": "you have",
    "you'll": "you will", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "we've": "we have",
    "they're": "they are", "they've": "they have",
    "that's": "that is", "what's": "what is",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "doesn't": "does not", "don't": "do not",
    "didn't": "did not", "wouldn't": "would not",
    "shouldn't": "should not", "couldn't": "could not",
    "mustn't": "must not",
}

# Negation trigger words — next word gets prefixed with "not_"
_NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "without", "barely", "hardly", "scarcely"}


def _expand_contractions(text):
    # type: (str) -> str
    """Expand contractions before further cleaning."""
    for contraction, expansion in _CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    return text


def _inject_negation_bigrams(tokens):
    # type: (list) -> list
    """
    Convert negation + next_word into a bigram token.
    Example: ["not", "happy", "today"] → ["not_happy", "today"]
    This preserves the semantic meaning through the pipeline.
    """
    result = []
    i = 0
    while i < len(tokens):
        if tokens[i] in _NEGATION_WORDS and i + 1 < len(tokens):
            # Create bigram with the following word
            bigram = "not_" + tokens[i + 1]
            result.append(bigram)
            i += 2   # skip both the negation and the next word
        else:
            result.append(tokens[i])
            i += 1
    return result


def _collapse_repeated_chars(text):
    # type: (str) -> str
    """
    Collapse repeated characters: 'loooove' → 'loove' (keep 2 max).
    Keeps enough for the stemmer to recognise the root.
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def clean_text(text):
    # type: (str) -> str
    """
    Full NLP preprocessing pipeline.

    Steps
    ─────
    1.  Lowercase
    2.  Expand contractions  (don't → do not)
    3.  Remove URLs
    4.  Remove @mentions
    5.  Remove # symbol (keep word)
    6.  Remove HTML entities
    7.  Remove non-ASCII / emoji
    8.  Collapse repeated chars (loooove → loove)
    9.  Remove punctuation & digits
    10. Collapse whitespace
    11. Tokenise + remove stopwords (preserving sentiment words)
    12. Inject negation bigrams  (not happy → not_happy)
    13. Porter stemming

    Parameters
    ----------
    text : str
        Raw tweet or comment text.

    Returns
    -------
    str
        Cleaned, stemmed string ready for TF-IDF vectorisation.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Expand contractions
    text = _expand_contractions(text)

    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 4. Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # 5. Remove # symbol but keep word
    text = re.sub(r"#", " ", text)

    # 6. Remove HTML entities
    text = re.sub(r"&[a-z]+;", " ", text)

    # 7. Remove non-ASCII
    text = text.encode("ascii", "ignore").decode("ascii")

    # 8. Collapse repeated characters
    text = _collapse_repeated_chars(text)

    # 9. Remove punctuation and digits
    text = re.sub(r"[^a-z\s]", " ", text)

    # 10. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 11. Tokenise + remove stopwords, keep words >= 2 chars
    tokens = [
        w for w in text.split()
        if w not in _FILTERED_STOPS and len(w) >= 2
    ]

    # 12. Inject negation bigrams BEFORE stemming
    tokens = _inject_negation_bigrams(tokens)

    # 13. Porter stemming (bigrams get stemmed on their suffix part)
    stemmed = []
    for tok in tokens:
        if tok.startswith("not_"):
            # Stem only the word after "not_"
            stemmed.append("not_" + _stemmer.stem(tok[4:]))
        else:
            stemmed.append(_stemmer.stem(tok))

    return " ".join(stemmed)
