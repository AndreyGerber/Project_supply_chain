import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from src.features.build_features import (
    generate_embeddings,
    get_structured_features
)
from src.utils.text_preprocessing import clean_text, add_structured_features
from src.models.predict_model import predict
MODEL_PATH = "models/model.joblib"

# =========================================
# LOAD MODEL ONCE
# =========================================
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
tfidf_pipeline = bundle["tfidf_pipeline"]
label_encoder = bundle["label_encoder"]
emb_version = bundle["embedding_version"]


# =========================================
# PREPROCESS
# =========================================
def preprocess(text: str) -> pd.DataFrame:

    df = pd.DataFrame({
        "review_text": [text],
        "supplier_response": [None],
        "verified": [0]
    })

    # alles vor Feature Engineering setzen
    df["review_text_en"] = df["review_text"]
    df["review_text_clean_en"] = df["review_text"].apply(clean_text)
    df["review_text_clean_light"] = df["review_text"].str.lower()

    # Safety Check 
    if "verified" not in df.columns:
        df["verified"] = 0
    df["verified"] = df["verified"].fillna(0).astype(int)

    df = add_structured_features(df)

    return df

# =========================================
# PREDICT (UI USE)
# =========================================
def predict_explain(text: str):

    df = preprocess(text)

    X_tfidf = tfidf_pipeline.transform(df["review_text_clean_en"])
    X_emb = generate_embeddings(df, version=emb_version)
    X_struct = get_structured_features(df)

    X = hstack([
        X_tfidf,
        csr_matrix(X_emb),
        csr_matrix(X_struct)
    ])

    pred_encoded = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    pred = label_encoder.inverse_transform([pred_encoded])[0]

    return {
        "prediction": int(pred),
        "probabilities": proba,
        "tfidf_vector": X_tfidf,
        "features": X
    }


# =========================================
# LOCAL EXPLANATION (FOR STREAMLIT)
# =========================================
def explain_local(model_input):

    X = model_input["features"]

    base_proba = model.predict_proba(X)[0]
    base_class = np.argmax(base_proba)

    X_np = X.toarray()
    importance = []

    for i in range(X_np.shape[1]):

        X_tmp = X_np.copy()
        X_tmp[:, i] = 0

        diff = base_proba[base_class] - model.predict_proba(X_tmp)[0][base_class]
        importance.append(diff)

    return np.array(importance)


# =========================================
# TF-IDF TOP WORDS
# =========================================
def get_tfidf_words(model_input, n=15):

    feature_names = tfidf_pipeline.get_feature_names_out()

    vec = model_input["tfidf_vector"].toarray().flatten()

    idx = np.argsort(vec)[::-1][:n]

    return pd.DataFrame({
        "word": [feature_names[i] for i in idx],
        "score": vec[idx]
    })

def predict_rating(text: str):
    
    result = predict([text])

    return int(result)