# SentimentAI — AI Based Social Media Sentiment Analysis

> ** AI Project** | NLP + Machine Learning | Python + Flask

---
# Team Members

| S.No | Name |
|------|--------------------------|
| 1 | K. Hema Lalitha Devi |
| 2 | T. Priyanka |
| 3 | K. Mahesh |
| 4 | N. Pushpak |

**Department:** Artificial Intelligence & Machine Learning

**University:** Aditya University, Surampalem

**Project Type:** Third Year Group Project

**Academic Year:** 2026–2027

## Project Overview

SentimentAI is a full-stack web application that performs real-time sentiment
analysis on social media text using Natural Language Processing (NLP) and
Machine Learning. It classifies any tweet, comment, or review as
**Positive** or **Negative** with a confidence score.

Trained on the **Sentiment140 dataset** (50,000 tweets), the model achieves
approximately **85–88% accuracy** using Logistic Regression with TF-IDF features.

---

## Features

| Feature | Description |
|---|---|
| Single Prediction | Paste any text and get instant sentiment + confidence % |
| Bulk CSV Upload | Upload thousands of comments and analyse them all at once |
| Visual Dashboard | Pie chart, bar chart, top words, recent predictions table |
| Download Report | Export all results as a CSV file |
| History | All predictions saved to SQLite database |
| Responsive UI | Works on desktop, tablet, and mobile |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| Web Framework | Flask 3.0 |
| Machine Learning | Scikit-learn (Logistic Regression) |
| NLP | NLTK (stopwords, Porter stemmer) |
| Feature Extraction | TF-IDF Vectorizer (75K vocab, bigrams) |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| Database | SQLite (built-in) |
| Frontend | HTML5, CSS3, Bootstrap 5, JavaScript |
| Charts | Chart.js 4 |
| Icons | Font Awesome 6 |

---

## Folder Structure

```
sentiment_analysis/
├── app.py                      # Flask app — all routes and logic
├── requirements.txt            # Python dependencies
├── sentiment_app.db            # SQLite database (auto-created)
│
├── data/
│   └── sentiment.csv           # Sentiment140 dataset (you provide)
│
├── ml/
│   ├── train_model.py          # ML training script (run once)
│   └── models/
│       ├── sentiment_model.pkl     # Trained model (auto-created)
│       └── tfidf_vectorizer.pkl    # TF-IDF vectorizer (auto-created)
│
├── utils/
│   ├── preprocessor.py         # Text cleaning + NLP pipeline
│   └── visualizer.py           # Chart data + word analytics
│
├── templates/
│   ├── base.html               # Master layout (navbar, footer)
│   ├── index.html              # Home + single predict + CSV upload
│   └── dashboard.html          # Charts + statistics + history table
│
└── static/
    ├── css/
    │   └── style.css           # Full glassmorphism UI styles
    └── js/
        └── script.js           # Drop-zone, animations, spinners
```

---

## Dataset

**Sentiment140** — Stanford Twitter Sentiment Dataset

- 1.6 million labelled tweets
- Labels: `0` = Negative, `4` = Positive
- This project uses the **first 50,000 rows**
- Download: https://www.kaggle.com/datasets/kazanova/sentiment140
- Place the file at: `data/sentiment.csv`
- The CSV has no header row; columns are:
  `target, id, date, query, user, text`

---

## AI Workflow

```
Raw Tweet Text
      │
      ▼
[1] Lowercase + Expand Contractions (don't → do not)
      │
      ▼
[2] Remove URLs, @mentions, #symbols, HTML entities
      │
      ▼
[3] Collapse repeated chars (loooove → loove)
      │
      ▼
[4] Remove punctuation and digits
      │
      ▼
[5] Remove stopwords (preserve negations + sentiment words)
      │
      ▼
[6] Inject negation bigrams (not happy → not_happy)
      │
      ▼
[7] Porter Stemming (running → run, absolutely → absolut)
      │
      ▼
[8] TF-IDF Vectorization (75K features, unigrams + bigrams)
      │
      ▼
[9] Logistic Regression → Positive / Negative + Confidence %
```

---

## Quick Start (Windows)

```bat
:: 1. Create virtual environment
python -m venv venv
venv\Scripts\activate

:: 2. Install dependencies
pip install -r requirements.txt

:: 3. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

:: 4. Place dataset
:: Copy sentiment.csv into the data\ folder

:: 5. Train the model (run ONCE — takes 2-4 minutes)
python ml\train_model.py

:: 6. Start the app
python app.py

:: 7. Open browser
:: http://localhost:5000
```

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Home page |
| POST | `/predict` | Single text prediction |
| POST | `/upload_csv` | Bulk CSV analysis |
| GET | `/dashboard` | Analytics dashboard |
| GET | `/download_csv` | Export results as CSV |
| GET | `/clear_session` | Reset dashboard to all-time history |
| GET | `/api/stats` | JSON — overall statistics |
| GET | `/api/history?n=20` | JSON — last N predictions |

---

## Model Performance

| Metric | Value |
|---|---|
| Training samples | 40,000 |
| Test samples | 10,000 |
| Accuracy | ~85–88% |
| Vectorizer | TF-IDF (75K vocab, bigrams) |
| Algorithm | Logistic Regression (saga solver) |
| Class weighting | Balanced |

---

## Contributors

 B.Tech / BCA / MCA AI Project
Department of Computer Science & Engineering

---

## License

For academic and educational purposes only.
Dataset credit: Go, A., Bhayani, R. & Huang, L. (2009). Sentiment140.
---

# Developed By

**AI Social Media Sentiment Analysis**

Developed by:

- K. Hema Lalitha Devi
- T. Priyanka
- K. Mahesh
- N. Pushpak

Department of Artificial Intelligence & Machine Learning

Aditya University, Surampalem

Academic Year: 2026–2027

---
