import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import random
import hashlib
import joblib

from src.utils.text_preprocessing import clean_text, add_structured_features
from src.data.load_data import load_raw_data
from src.features.store_feature import FeatureStore
from src.utils.data_cleaning import clean_raw_data

# ---------------- Paths ----------------
PROCESSED_PATH = Path(__file__).resolve().parent.parent.parent / "data/processed/reviews_clean.csv"
FEATURE_PATH = Path(__file__).resolve().parent.parent.parent / "data/features"

store = FeatureStore(FEATURE_PATH)

# ---------------- Global Seed ----------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- Embedding Model ----------------
EMBEDDING_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# ---------------- HASH ----------------
def generate_feature_hash(df: pd.DataFrame, version: str) -> str:
    """
    Stabiler Hash über relevante Inhalte
    """
    content = (
        df["review_text"].astype(str) +
        df["supplier_response"].astype(str) +
        df["verified"].astype(str)
    ).str.cat(sep=" ")

    raw = content + version
    return hashlib.md5(raw.encode()).hexdigest()[:10]


# ---------------- PREPROCESS ----------------
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_text_clean"] = df["review_text"].astype(str).apply(clean_text)
    df["rating"] = df["rating_svg"].astype(float)

    df = add_structured_features(df)

    return df


# ---------------- TF-IDF ----------------
def generate_tfidf(df: pd.DataFrame, version="v1", max_features: int = 5000):
    """
    Gibt UNFITTED Pipeline zurück (wichtig für sklearn Pipeline!)
    Caching nur für fitted Pipeline optional extern
    """

    text_features = "review_text_clean"
    structured_features = ["review_length", "verified", "has_response"]

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2)
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", tfidf, text_features),
            ("struct", StandardScaler(), structured_features)
        ]
    )

    return preprocessor, df


# ---------------- EMBEDDINGS ----------------
def generate_embeddings(df: pd.DataFrame, version="v1", use_clean_text=True):
    """
    Multilingual Embeddings + strukturierte Features
    """

    feature_hash = generate_feature_hash(df, version)
    cache_name = f"embeddings_{feature_hash}"

    cached = store.load_embeddings(cache_name)
    if cached is not None:
        return cached

    text_col = "review_text_clean" if use_clean_text else "review_text"
    texts = df[text_col].astype(str).tolist()

    print(f"⚡ Generating embeddings ({len(texts)} samples)...")

    text_embeddings = EMBEDDING_MODEL.encode(
        texts,
        show_progress_bar=True
    )

    structured = df[["review_length", "verified", "has_response"]].values

    features = np.hstack([structured, text_embeddings])

    store.save_embeddings(cache_name, features)

    return features


# ---------------- SAVE ----------------
def save_processed(df: pd.DataFrame):
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_PATH}")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    df_raw = load_raw_data()
    df_cleaned = clean_raw_data(df_raw)
    df_processed = preprocess_dataframe(df_cleaned)
    save_processed(df_processed)