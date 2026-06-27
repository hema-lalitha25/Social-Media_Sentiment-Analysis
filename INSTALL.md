# Installation Guide — SentimentAI (Windows)

## Prerequisites

| Tool | Version | Download |
|---|---|---|
| Python | 3.8 – 3.12 | https://python.org/downloads |
| pip | (bundled) | — |
| VS Code | Latest | https://code.visualstudio.com |
| Git | Optional | https://git-scm.com |

---

## Step 1 — Extract the Project

Unzip `sentiment_analysis.zip` to any folder, e.g.:
```
C:\Projects\sentiment_analysis\
```

Open this folder in VS Code:
```
File → Open Folder → select sentiment_analysis
```

---

## Step 2 — Open Terminal in VS Code

```
Terminal → New Terminal   (or Ctrl + `)
```

---

## Step 3 — Create Virtual Environment

```bat
python -m venv venv
```

Activate it:
```bat
venv\Scripts\activate
```

You should see `(venv)` appear at the start of the terminal prompt.

---

## Step 4 — Install Dependencies

```bat
pip install -r requirements.txt
```

This installs: Flask, scikit-learn, pandas, numpy, nltk, joblib, werkzeug.

---

## Step 5 — Download NLTK Data

```bat
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## Step 6 — Download the Dataset

1. Go to: https://www.kaggle.com/datasets/kazanova/sentiment140
2. Download `training.1600000.processed.noemoticon.csv`
3. **Rename** it to `sentiment.csv`
4. Place it inside the `data\` folder:

```
sentiment_analysis\
  data\
    sentiment.csv    ← HERE
```

> **Note:** The file has no header row. Do not add one.

---

## Step 7 — Train the Model

```bat
python ml\train_model.py
```

This will:
- Load 50,000 rows from `data\sentiment.csv`
- Clean and preprocess the text
- Train Logistic Regression with TF-IDF
- Save `ml\models\sentiment_model.pkl`
- Save `ml\models\tfidf_vectorizer.pkl`
- Print accuracy (~85–88%) in the terminal

**Expected time:** 2–5 minutes depending on your CPU.

---

## Step 8 — Start the Web App

```bat
python app.py
```

Expected output:
```
  Database initialised.
  Model and vectorizer loaded successfully.
 * Running on http://0.0.0.0:5000
 * Debug mode: on
```

---

## Step 9 — Open in Browser

```
http://localhost:5000
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: flask` | Run `pip install -r requirements.txt` |
| `Model files not found` | Run `python ml\train_model.py` first |
| `FileNotFoundError: sentiment.csv` | Place dataset in `data\sentiment.csv` |
| `LookupError: stopwords` | Run `python -c "import nltk; nltk.download('stopwords')"` |
| Port 5000 in use | Change port in `app.py`: `app.run(port=5001)` |
| `venv\Scripts\activate` fails | Run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## Folder Structure After Full Setup

```
sentiment_analysis\
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── INSTALL.md
├── sentiment_app.db          ← created on first run
├── data\
│   └── sentiment.csv         ← YOU place this here
├── ml\
│   ├── train_model.py
│   └── models\
│       ├── sentiment_model.pkl     ← created by train_model.py
│       └── tfidf_vectorizer.pkl    ← created by train_model.py
├── utils\
│   ├── __init__.py
│   ├── preprocessor.py
│   └── visualizer.py
├── templates\
│   ├── base.html
│   ├── index.html
│   └── dashboard.html
├── static\
│   ├── css\
│   │   └── style.css
│   └── js\
│       └── script.js
└── uploads\                  ← CSV uploads stored temporarily
```
