import joblib
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split

from src.analysis.feature_analysis import (
    run_group_permutation_importance,
    plot_group_importance
)

from src.data.load_data import load_processed_data
from src.features.build_features import (
    generate_embeddings,
    get_structured_features
)


# =========================================
# 📁 PATHS
# =========================================
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models/model.joblib"
OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "analysis_results"


# =========================================
# 🔧 BUILD TEST DATA (IDENTISCH ZU TRAINING)
# =========================================
def build_test_data():
    df = load_processed_data()
    df = df.dropna(subset=["review_text_clean_en", "rating"])

    X_text = df["review_text_clean_en"]
    y = df["rating"].astype(int)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return df, X_test_text, y_test


# =========================================
# 🚀 MAIN PIPELINE
# =========================================
def main():

    print("📦 Loading model...")
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    tfidf_pipeline = bundle["tfidf_pipeline"]
    label_encoder = bundle["label_encoder"]

    print("📊 Building test data...")
    df, X_test_text, y_test = build_test_data()

    df_test = df.loc[X_test_text.index]

    print("🔧 Generating features...")

    # --- TF-IDF ---
    X_test_tfidf = tfidf_pipeline.transform(X_test_text)

    # --- Embeddings ---
    X_test_emb = generate_embeddings(df_test, version="v1")

    # --- Structured ---
    X_test_struct = get_structured_features(df_test)

    # --- Combine ---
    X_test = hstack([
        X_test_tfidf,
        csr_matrix(X_test_emb),
        csr_matrix(X_test_struct)
    ])

    # --- Encode Labels ---
    y_test = label_encoder.transform(y_test)

    print("📊 Running feature analysis...")

    results = run_group_permutation_importance(
        model=model,
        X_test=X_test,
        y_test=y_test,
        tfidf_dim=5000,
        emb_dim=384,
        struct_dim=4,
        n_runs=5
    )

    # =========================================
    # 💾 SAVE RESULTS
    # =========================================
    OUTPUT_PATH.mkdir(exist_ok=True)

    df_results = pd.DataFrame(results).T
    df_results.to_csv(OUTPUT_PATH / "feature_importance.csv")

    print(f"💾 Results saved to {OUTPUT_PATH}")

    # Plot anzeigen
    plot_group_importance(results)


if __name__ == "__main__":
    main()