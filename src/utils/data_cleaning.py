# src/utils/data_cleaning.py

import pandas as pd


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bereinigt Rohdaten:
    - Entfernt Duplikate
    - Entfernt leere Texte
    - Entfernt sehr kurze Texte
    - Entfernt ungültige Ratings
    """

    df = df.copy()

    print(f"Initial shape: {df.shape}")

    # ---------------- Remove duplicates ----------------
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")

    # ---------------- Remove empty review_text ----------------
    df["review_text"] = df["review_text"].astype(str)
    df = df[df["review_text"].str.strip().astype(bool)]
    print(f"After removing empty texts: {df.shape}")

    # ---------------- Remove very short texts ----------------
    df = df[df["review_text"].apply(lambda x: len(x.split()) >= 3)]
    print(f"After removing short texts: {df.shape}")

    # ---------------- Clean supplier_response ----------------
    if "supplier_response" in df.columns:
        df["supplier_response"] = df["supplier_response"].fillna("")

    # ---------------- Validate rating ----------------
    if "rating_svg" in df.columns:
        df = df[df["rating_svg"].notna()]
        df = df[df["rating_svg"].between(1, 5)]
        print(f"After rating cleaning: {df.shape}")

    return df