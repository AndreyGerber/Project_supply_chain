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
from src.utils.text_translation import ReviewTranslator

# ---------------- Paths ----------------
PROCESSED_PATH = Path(__file__).resolve().parent.parent.parent / "data/processed/reviews_processed.csv"
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
    translator = ReviewTranslator()
    
    df = df.copy()
    #df["review_text_clean"] = df["review_text"].astype(str).apply(clean_text)
    df["rating"] = df["rating"].astype(float)
    df = translator.process_dataframe_for_translation(df)
    df["review_text_clean_en"] = df["review_text_en"].apply(clean_text)
    df["review_text_clean_light"] = df["review_text_en"].str.lower()
    df = add_structured_features(df)

    return df


# ---------------- TF-IDF ----------------
def generate_tfidf(df: pd.DataFrame, version="v2", max_features: int = 5000):
    """
    Gibt UNFITTED Pipeline zurück (wichtig für sklearn Pipeline!)
    Caching nur für fitted Pipeline optional extern
    """

    text_features = "review_text_clean_en"
    structured_features = ["review_length","sentiment", "verified", "has_negation",]

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

#------------------Tf-IDF in Pipeline (nur text)----------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

def get_tfidf_pipeline(max_features: int = 5000):
    """
    TF-IDF Pipeline (nur Text, kein struct!)
    """

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2)
    )

    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("scaler", MaxAbsScaler())
    ])

    return pipeline

#---------------- Embeddings Pipeline (nur text) ----------------

def generate_embeddings(df: pd.DataFrame, version="v4"):
    """
    Nur Embeddings – KEINE strukturierten Features!
    """

    feature_hash = generate_feature_hash(df, version)
    cache_name = f"embeddings_{feature_hash}"

    cached = store.load_embeddings(cache_name)
    if cached is not None:
        return cached

    texts = df["review_text_clean_light"].astype(str).tolist()

    embeddings = np.array(
        EMBEDDING_MODEL.encode(texts, show_progress_bar=True),
        dtype=np.float32
    )

    store.save_embeddings(cache_name, embeddings)

    return embeddings

from sklearn.decomposition import PCA

def reduce_embeddings(X, n_components=100):
    pca = PCA(n_components=n_components, random_state=SEED)
    return pca.fit_transform(X)

#---------nur Strukturierte Features----------------
def get_structured_features(df: pd.DataFrame, scale=True):
    """
    Nur strukturierte Features
    """

    struct_cols = ["review_length", "verified", "sentiment", "has_negation"]
    X = df[struct_cols].values.astype(np.float32)

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X



#-----------Hybride Features (TF-IDF + Embeddings + Structured)----------------
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import csr_matrix

def build_hybrid_features(df: pd.DataFrame, use_pca=False):
    """
    Kombiniert TF-IDF + Embeddings + Structured Features
    """

    # ---------------- TF-IDF ----------------
    tfidf_pipeline = get_tfidf_pipeline()
    X_tfidf = tfidf_pipeline.fit_transform(df["review_text_clean_en"])

    # ---------------- Embeddings ----------------
    X_emb = generate_embeddings(df,)

    if use_pca:
        X_emb = reduce_embeddings(X_emb, n_components=100)

    # ---------------- Structured ----------------
    X_struct = get_structured_features(df)

    # ---------------- Sparse Handling ----------------
    X_emb_sparse = csr_matrix(X_emb)
    X_struct_sparse = csr_matrix(X_struct)

    # ---------------- Combine ----------------
    X_final = sparse_hstack([
        X_tfidf,
        X_emb_sparse,
        X_struct_sparse
    ])

    return X_final, tfidf_pipeline



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