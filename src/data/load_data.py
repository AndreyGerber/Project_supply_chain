import pandas as pd
from pathlib import Path
from typing import Tuple



#Path anpassen
RAW_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "trustpilot_reviews_production.json"
PROCESSED_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "reviews_clean.csv"

def load_raw_data() -> pd.DataFrame:
    """
    Load the raw Trustpilot data.
    Returns a DataFrame.
    """
    # Anpassen je nach Format (CSV/JSON)
    if RAW_PATH.suffix == ".csv":
        df = pd.read_csv(RAW_PATH)
    else:
        df = pd.read_json(RAW_PATH)
    return df

def load_processed_data() -> pd.DataFrame:
    """
    Load already cleaned/processed data.
    No such file->> run build_features.py first to create it.
    """
    if PROCESSED_PATH.exists():
        print("✅ Loading processed dataset...")
        return pd.read_csv(PROCESSED_PATH)

    print("⚠️ Processed data not found. Running cleaning pipeline...")
    from src.features.build_features import save_processed # import hier, um circular imports zu vermeiden
    from src.features.build_features import preprocess_dataframe  
    from src.utils.data_cleaning import clean_raw_data

    #laden
    df_raw = load_raw_data()
    #bereinigen
    df_cleaned = clean_raw_data(df_raw)
    df_processed = preprocess_dataframe(df_cleaned)
    # speichern
    save_processed(df_processed)
    print("✅ Processed dataset created and saved.")

    return pd.read_csv(PROCESSED_PATH)

def get_data(use_processed: bool = True) -> pd.DataFrame:
    """
    Utility function to get either raw or processed data.
    """
    if use_processed:
        return load_processed_data()
    else:
        return load_raw_data()

if __name__ == "__main__":
    df = get_data()
    print(df.head())
    print(f"Dataset size: {df.shape}")