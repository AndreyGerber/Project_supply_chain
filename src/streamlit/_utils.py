"""Shared helpers for all pages: data loading, text preprocessing and the model pipeline.

Every page loads its data through the cached functions below, so each page works on
its own (no need to visit the pages in order) and expensive steps run only once.
"""
import json
import random
import re
import time
from pathlib import Path

import nltk
import pandas as pd
import scipy.sparse as sp
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data/raw/trustpilot_reviews_production.json"
PROCESSED_PATH = ROOT / "data/processed/reviews_processed.csv"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# --- Shared constants ---
LABELS = ["Low (1-2 ⭐)", "Mid (3-4 ⭐)", "High (5 ⭐)"]
LABEL_COLORS = {"High (5 ⭐)": "#00CC96", "Mid (3-4 ⭐)": "#AB63FA", "Low (1-2 ⭐)": "#EF553B"}
RATING_LABELS = ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]
REPLY_PATTERN = r"^Reply from"
TARGET_PER_CLASS = 2000

MODEL_PARAMS = {
    "base": dict(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "tuned": dict(n_estimators=300, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42),
}

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none", "but",
    "dont", "doesnt", "didnt", "wasnt", "werent", "havent", "hasnt", "hadnt",
    "isnt", "arent", "wouldnt", "shouldnt", "couldnt", "cant", "cannot",
}
SYMBOL_STOPS = {
    ",", ".", "``", "@", "*", "(", ")", "...", "!", "?", "-", "_", ">", "<",
    ":", "/", "=", "--", "©", "~", ";", "\\", "\\\\",
}


# =========================
# Data loading
# =========================
@st.cache_data
def load_raw() -> pd.DataFrame:
    """Scraped reviews as they came from Trustpilot."""
    with open(RAW_PATH, encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


@st.cache_data
def load_processed() -> pd.DataFrame:
    """Processed reviews, with the English translation as the main text column."""
    df = pd.read_csv(PROCESSED_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "review_text_en" in df.columns:
        df = df.drop(columns=["review_text"], errors="ignore").rename(columns={"review_text_en": "review_text"})
    if "review_text_clean_en" in df.columns:
        df = df.rename(columns={"review_text_clean_en": "review_text_clean_advanced"})
    return df


def reply_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows that are only an automated company reply."""
    return df["review_text"].str.contains(REPLY_PATTERN, na=False, case=False, regex=True)


def remove_redundant(df: pd.DataFrame) -> pd.DataFrame:
    """Drop company replies and duplicate review texts (keeps the first occurrence)."""
    return df[~reply_mask(df)].drop_duplicates(subset=["review_text"], keep="first")


@st.cache_data
def load_ml_data() -> pd.DataFrame:
    """Result of the Preprocessing page: unique reviews with the model features."""
    return remove_redundant(load_processed())[["verified", "review_text", "rating"]].copy()


# =========================
# Text preprocessing
# =========================
@st.cache_resource
def _stop_words() -> frozenset:
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab", "corpora/stopwords"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[1], quiet=True)
    return frozenset((set(stopwords.words("english")) - NEGATION_WORDS) | SYMBOL_STOPS)


def preprocess_text_full(text) -> str:
    """Lowercase -> keep letters only -> tokenize -> remove stopwords (but keep negations) -> join."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[’']", "", text.lower())     # "don't" -> "dont", so negations survive
    text = re.sub(r"[^a-z\s]", " ", text)        # space, not "", so "address,I" does not become "addressi"
    stop_words = _stop_words()
    return " ".join(w for w in word_tokenize(text) if w not in stop_words)


# =========================
# Extra features (from the Feature Engineering page)
# =========================
NEGATION_PATTERN = r"\b(not|no|never)\b"


@st.cache_resource
def _sentiment_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def extra_features(texts: pd.Series) -> pd.DataFrame:
    """Sentiment score (-1 to +1) and negation flag (0/1) for each review."""
    analyzer = _sentiment_analyzer()
    return pd.DataFrame({
        "sentiment": texts.apply(lambda t: analyzer.polarity_scores(t)["compound"] if isinstance(t, str) else 0.0),
        "has_negation": texts.str.contains(NEGATION_PATTERN, case=False, na=False).astype(float),
    }, index=texts.index)


# =========================
# Modelling pipeline
# =========================
def group_rating(rating) -> str:
    if rating <= 2:
        return LABELS[0]
    if rating <= 4:
        return LABELS[1]
    return LABELS[2]


def swap_two_words(text, rng: random.Random):
    """Swaps two random words to slightly vary the text without changing its meaning."""
    if not isinstance(text, str):
        return text
    words = text.split()
    if len(words) < 3:
        return text
    i, j = rng.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return " ".join(words)


def balance(train_df: pd.DataFrame):
    """Undersample 'High', oversample 'Mid' and 'Low' with word-swap augmentation."""
    rng = random.Random(42)
    frames, log = [], []
    for group in LABELS:
        subset = train_df[train_df["target_group"] == group]
        if subset.empty:
            continue
        if group == LABELS[2]:
            balanced = subset.sample(n=min(len(subset), TARGET_PER_CLASS), random_state=42)
            log.append((group, len(subset), len(balanced), "Undersampling"))
        elif len(subset) < TARGET_PER_CLASS:
            extras = subset.sample(n=TARGET_PER_CLASS - len(subset), replace=True, random_state=42)
            extras["review_text"] = extras["review_text"].apply(swap_two_words, rng=rng)
            balanced = pd.concat([subset, extras])
            log.append((group, len(subset), len(balanced), "Augmentation"))
        else:
            balanced = subset
        frames.append(balanced)
    return pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True), log


@st.cache_resource(show_spinner="Preparing the training data…")
def build_dataset() -> dict:
    """All steps from cleaning to the preprocessed train/test text. Returns intermediate results for display."""
    df = load_ml_data().copy()
    stats = {"exact_dups": int(df.duplicated().sum())}

    # Same text (ignoring case/whitespace) with the same rating counts as a duplicate
    normalized_dup = pd.concat([df["review_text"].astype(str).str.lower().str.strip(), df["rating"]], axis=1).duplicated()
    stats["normalized_dups"] = int(normalized_dup.sum())
    df = df[~normalized_dup]

    replies = reply_mask(df)
    stats["nans"] = int(df["review_text"].isna().sum())
    stats["replies"] = int(replies.sum())
    df = df[df["review_text"].notna() & df["rating"].notna() & ~replies].copy()
    df["target_group"] = df["rating"].apply(group_rating)

    X_train, X_test, y_train, y_test = train_test_split(
        df[["review_text", "verified"]], df["target_group"],
        test_size=0.2, random_state=42, stratify=df["target_group"],
    )

    train_df, balance_log = balance(X_train.assign(target_group=y_train))
    X_train_bal = train_df[["review_text", "verified"]].copy()
    X_train_bal["clean_review"] = X_train_bal["review_text"].apply(preprocess_text_full)
    X_test = X_test.copy()
    X_test["clean_review"] = X_test["review_text"].apply(preprocess_text_full)

    return {
        "df": df, "stats": stats,
        "y_train_split": y_train, "y_test": y_test,
        "train_balanced": train_df, "balance_log": balance_log,
        "X_train_text": X_train_bal, "y_train": train_df["target_group"], "X_test_text": X_test,
    }


# Two feature sets: plain TF-IDF (first and second try) and the enhanced set (third try)
FEATURE_SETS = {
    "tfidf": dict(ngram_range=(1, 1), max_features=5000, extra=False),
    "enhanced": dict(ngram_range=(1, 2), max_features=10000, extra=True),
}
# Which feature set and which hyperparameters each model uses
MODELS = {
    "base": ("tfidf", "base"),
    "tuned": ("tfidf", "tuned"),
    "enhanced": ("enhanced", "tuned"),
}


def _combine(tfidf_matrix, texts: pd.DataFrame, extra: bool):
    """TF-IDF columns + verified (+ sentiment and negation for the enhanced set)."""
    blocks = [tfidf_matrix, sp.csr_matrix(texts[["verified"]].values.astype(float))]
    if extra:
        blocks.append(sp.csr_matrix(extra_features(texts["review_text"]).values))
    return sp.hstack(blocks).tocsr()


@st.cache_resource(show_spinner=False)
def build_features(feature_set: str) -> dict:
    """Fits the TF-IDF vectorizer on the training text and builds the feature matrices."""
    data, cfg = build_dataset(), FEATURE_SETS[feature_set]
    tfidf = TfidfVectorizer(ngram_range=cfg["ngram_range"], max_features=cfg["max_features"])
    X_train = _combine(tfidf.fit_transform(data["X_train_text"]["clean_review"]), data["X_train_text"], cfg["extra"])
    X_test = _combine(tfidf.transform(data["X_test_text"]["clean_review"]), data["X_test_text"], cfg["extra"])
    return {"X_train": X_train, "X_test": X_test, "vectorizer": tfidf}


@st.cache_resource(show_spinner=False)
def train_model(name: str) -> dict:
    """Trains one of the models in MODELS once and keeps it in memory for all pages."""
    feature_set, params = MODELS[name]
    data, features = build_dataset(), build_features(feature_set)
    start = time.time()
    model = GradientBoostingClassifier(**MODEL_PARAMS[params])
    model.fit(features["X_train"], data["y_train"])
    return {
        "model": model,
        "y_pred": model.predict(features["X_test"]),
        "duration": time.time() - start,
    }


def vectorize_review(text: str, verified: bool, name: str):
    """Turns one new review into the same feature format the given model was trained on."""
    feature_set = MODELS[name][0]
    cleaned = preprocess_text_full(text)
    row = pd.DataFrame({"review_text": [text], "verified": [1.0 if verified else 0.0]})
    vector = _combine(build_features(feature_set)["vectorizer"].transform([cleaned]), row,
                      FEATURE_SETS[feature_set]["extra"])
    return cleaned, vector
