# src/utils/data_cleaning.py

import pandas as pd


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bereinigt Rohdaten:
    - Entfernt Duplikate
    - Entfernt leere Texte
    - Entfernt sehr kurze Texte
    - Bereinigt Datumsangaben (zu datetime)
    - Bereinigt Verifiziert-Status(zu bool)
    - Entfernt ungültige Ratings (muss float 1-5)
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

     # ---------------- Remove review_text starts with "Reply" ----------------
    df = df[~df["review_text"].str.startswith("Reply", na=False)]
    print(f"After removing Reply texts: {df.shape}")

    # ---------------- Clean supplier_response ----------------
    if "supplier_response" in df.columns:
        df["supplier_response"] = df["supplier_response"].fillna("")

    # ---------------- Clean Date ----------------
    if "date" in df.columns:
        df =clean_date_column(df)
        print(f"After date cleaning: {df['date'].isna().sum()} missing values")
        

    if "verified" in df.columns:
        df["verified"] = df["verified"].fillna(0).astype(int)
        df["verified"] = df["verified"].astype(bool)
        print(f"After verified cleaning: {df['verified'].isna().sum()} missing values")

    # ---------------- Validate rating ----------------
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df[df["rating"].notna()]
        df = df[df["rating"].between(1, 5)]
        print(f"After rating cleaning: {df.shape}")

    return df

# ---------------- Clean Date ----------------
def clean_date_column(df):
    if pd.api.types.is_datetime64_any_dtype(df['date']):
        return df  # schon sauber

    parsed = pd.to_datetime(df['date'], errors='coerce', utc=True)

    numeric = pd.to_numeric(df['date'], errors='coerce')
    mask = parsed.isna() & numeric.notna()

    parsed.loc[mask] = pd.to_datetime(numeric[mask], unit='ms', errors='coerce', utc=True)

    df['date'] = parsed
    return df