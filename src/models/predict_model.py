import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.build_features import generate_embeddings
from src.utils.text_preprocessing import clean_text, add_structured_features

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models/model.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


def predict(texts: list[str], feature_type="tfidf") -> list:
    model = load_model()

    df = pd.DataFrame({
        "review_text": texts,
        "supplier_response": [None] * len(texts),
        "verified": [0] * len(texts)
    })

    df["review_text_clean"] = df["review_text"].apply(clean_text)
    df = add_structured_features(df)

    if feature_type == "tfidf":
        preds = model.predict(df,version="v1")

    elif feature_type == "embedding":
        X = generate_embeddings(df, version="v1")
        preds = model.predict(X)

    else:
        raise ValueError("Unknown feature type")

    return preds


if __name__ == "__main__":
    sample = [
        "Great product, fast delivery!",
        "Terrible service, very disappointed."
    ]

    print(predict(sample, feature_type="embedding"))