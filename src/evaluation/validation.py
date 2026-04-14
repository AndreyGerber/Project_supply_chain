# src/evaluation/validation.py

import pandas as pd
from sklearn.model_selection import cross_val_score


def cross_validate_model(pipeline, X, y):
    """
    Cross Validation für robuste Performance-Schätzung
    """
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1_weighted")
    
    print("\nCross Validation Scores:", scores)
    print(f"Mean F1 Score: {scores.mean():.4f}")


def check_class_balance(y):
    """
    Prüft Klassenverteilung
    """
    print("\nClass Distribution:")
    print(y.value_counts(normalize=True))


def check_data_leakage(df: pd.DataFrame):
    """
    Einfacher Leakage Check über Korrelationen
    """
    print("\nChecking for data leakage...")
    
    if "rating" in df.columns:
        corr = df.corr(numeric_only=True)
        print(corr["rating"].sort_values(ascending=False))


def test_edge_cases(pipeline):
    """
    Testet Modell auf ungewöhnliche Inputs
    """
    samples = [
        "Worst experience ever",
        "Absolutely amazing service",
        "ok",
        "",
        "12345"
    ]

    df = pd.DataFrame({
        "review_text": samples,
        "supplier_response": [None]*len(samples),
        "verified": [0]*len(samples),
        "review_text_clean": samples
    })

    preds = pipeline.predict(df)

    print("\nEdge Case Predictions:")
    for s, p in zip(samples, preds):
        print(f"{s} -> {p}")