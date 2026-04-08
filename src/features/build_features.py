import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# interne Funktionen importieren
from src.utils.text_preprocessing import clean_text, add_structured_features
from src.data.load_data import load_raw_data

PROCESSED_PATH = Path(__file__).resolve().parent.parent.parent / "data/processed/reviews_clean.csv"

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Komplette Preprocessing-Pipeline:
    - Text cleaning
    - Rating numerisch
    - Strukturierte Features
    """
    df = df.copy()
    # Text cleaning
    df["review_text_clean"] = df["review_text"].astype(str).apply(clean_text)
    #Rating numerisch
    df['rating'] = df['rating_svg'].astype(float)
    # Zusätzliche Features
    df = add_structured_features(df)
    
    return df

def generate_tfidf(df: pd.DataFrame, max_features: int = 5000) -> (Pipeline, pd.DataFrame):
    """
    Erstellt TF-IDF Features und kombiniert sie mit strukturierten Features
    """
    text_features = "review_text_clean"
    structured_features = ["review_length", "verified", "has_response"]
    
    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    
    # Spalten-Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", tfidf, text_features),
            ("struct", StandardScaler(), structured_features)
        ]
    )
    
    return preprocessor, df

def save_processed(df: pd.DataFrame):
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_PATH}")

if __name__ == "__main__":
    df_raw = load_raw_data()
    df_processed = preprocess_dataframe(df_raw)
    save_processed(df_processed)