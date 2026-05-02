import joblib
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

from src.features.build_features import (
    generate_embeddings,
    get_structured_features
)

from src.utils.text_preprocessing import clean_text, add_structured_features

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models/model.joblib"


# =========================================
# LOAD MODEL
# =========================================
def load_model():
    bundle = joblib.load(MODEL_PATH)

    return (
        bundle["model"],
        bundle["tfidf_pipeline"],
        bundle["label_encoder"],
        bundle["embedding_version"]
    )


# =========================================
# PREPROCESS
# =========================================
def preprocess_input(texts: list[str]) -> pd.DataFrame:

    df = pd.DataFrame({
        "review_text": texts,
        "supplier_response": [None] * len(texts),
        "verified": [0] * len(texts)
    })

    df["review_text_clean_en"] = df["review_text"].apply(clean_text)
    df["review_text_clean_light"] = df["review_text"].str.lower()
    df["review_text_en"] = df["review_text"]
    df = add_structured_features(df)

    return df


# =========================================
# PREDICT
# =========================================
def predict(texts: list[str]):

    model, tfidf_pipeline, le, emb_version = load_model()

    df = preprocess_input(texts)

    # ---- TF-IDF ----
    X_tfidf = tfidf_pipeline.transform(df["review_text_clean_en"])

    # ---- Embeddings ----
    X_emb = generate_embeddings(df, version=emb_version)

    # ---- Structured ----
    X_struct = get_structured_features(df)

    # ---- Combine ----
    X = hstack([
        X_tfidf,
        csr_matrix(X_emb),
        csr_matrix(X_struct)
    ])

    preds_encoded = model.predict(X)
    preds = le.inverse_transform(preds_encoded)

    return preds.tolist()


if __name__ == "__main__":

    sample = [
        "Great product, fast delivery!",
        "Okay experience, but could be better.",
        "Terrible service, very disappointed."
    ]

    print(predict(sample))
