import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, cohen_kappa_score
)
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from scipy.sparse import hstack, csr_matrix

from src.data.load_data import load_processed_data
from src.features.build_features import (
    get_tfidf_pipeline,
    generate_embeddings,
    get_structured_features
)

RESULT_PATH = Path("data/experiments/full_study.csv")
PRED_PATH = Path("data/experiments/predictions.csv")


# =========================================
# Ordinal Weights
# =========================================
def create_ordinal_weights(y):
    counts = np.bincount(y)
    class_weights = 1.0 / counts
    base = class_weights[y]
    ordinal = np.abs(y - np.median(y)) + 1
    return base * ordinal


# =========================================
# Metrics
# =========================================
def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "mae": mean_absolute_error(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred)
    }


# =========================================
# Feature Builder
# =========================================
def build_hybrid(df_train, df_test, seed):

    tfidf = get_tfidf_pipeline()
    df_train["review_text_clean_en"] = df_train["review_text_clean_en"].fillna("")
    df_test["review_text_clean_en"] = df_test["review_text_clean_en"].fillna("")
    
    X_train_tfidf = tfidf.fit_transform(df_train["review_text_clean_en"])
    X_test_tfidf = tfidf.transform(df_test["review_text_clean_en"])

    X_train_emb = generate_embeddings(df_train, version=f"trainHyb_{seed}")
    X_test_emb = generate_embeddings(df_test, version=f"testHyb_{seed}")

    X_train_struct = get_structured_features(df_train)
    X_test_struct = get_structured_features(df_test)

    X_train = hstack([
        X_train_tfidf,
        csr_matrix(X_train_emb),
        csr_matrix(X_train_struct)
    ])

    X_test = hstack([
        X_test_tfidf,
        csr_matrix(X_test_emb),
        csr_matrix(X_test_struct)
    ])

    return X_train, X_test


# =========================================
# MAIN STUDY
# =========================================
def run_study(seeds=[42, 1337, 2024]):

    df = load_processed_data()
    results = []
    predictions = []

    for seed in seeds:

        print(f"\n🚀 Seed {seed}")

        df_train, df_test = train_test_split(
            df,
            test_size=0.2,
            stratify=df["rating"],
            random_state=seed
        )

        le = LabelEncoder()

        y_train = le.fit_transform(df_train["rating"].astype(int))
        y_test = le.transform(df_test["rating"].astype(int))

        # =========================================
        # EXP1: Embedding + Struct
        # =========================================
        X_train_emb = generate_embeddings(df_train, version=f"train_{seed}")
        X_test_emb = generate_embeddings(df_test, version=f"test_{seed}")

        X_train_struct = get_structured_features(df_train)
        X_test_struct = get_structured_features(df_test)

        X_train_exp1 = np.hstack([X_train_emb, X_train_struct])
        X_test_exp1 = np.hstack([X_test_emb, X_test_struct])

        model = XGBClassifier(eval_metric="mlogloss")
        model.fit(X_train_exp1, y_train)

        y_pred = model.predict(X_test_exp1)

        res = evaluate(y_test, y_pred)
        res.update({"experiment": "Exp1", "seed": seed})
        results.append(res)

        predictions.append(pd.DataFrame({
            "y_true": le.inverse_transform(y_test),
            "y_pred": le.inverse_transform(y_pred),
            "experiment": "Exp1",
            "seed": seed
        }))

        # =========================================
        # EXP2: Exp1 + GridSearch
        # =========================================
        param_grid = {
            "n_estimators": [100, 300],
            "max_depth": [4, 6]
        }

        grid = GridSearchCV(
            XGBClassifier(eval_metric="mlogloss"),
            param_grid,
            cv=3,
            scoring="f1_weighted",
            n_jobs=-1
        )

        grid.fit(X_train_exp1, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test_exp1)  # ✅ FIX

        res = evaluate(y_test, y_pred)
        res.update({"experiment": "Exp2", "seed": seed})
        results.append(res)

        predictions.append(pd.DataFrame({
            "y_true": le.inverse_transform(y_test),
            "y_pred": le.inverse_transform(y_pred),
            "experiment": "Exp2",
            "seed": seed
        }))

        # =========================================
        # EXP3: Hybrid + GridSearch
        # =========================================
        X_train, X_test = build_hybrid(df_train, df_test, seed)

        grid = GridSearchCV(
            XGBClassifier(eval_metric="mlogloss"),
            param_grid,
            cv=3,
            scoring="f1_weighted",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)

        res = evaluate(y_test, y_pred)
        res.update({"experiment": "Exp3", "seed": seed})
        results.append(res)

        predictions.append(pd.DataFrame({
            "y_true": le.inverse_transform(y_test),
            "y_pred": le.inverse_transform(y_pred),
            "experiment": "Exp3",
            "seed": seed
        }))

        # =========================================
        # EXP4: Exp3 + Ordinal
        # =========================================
        weights = create_ordinal_weights(y_train)

        model = XGBClassifier(eval_metric="mlogloss")
        model.fit(X_train, y_train, sample_weight=weights)

        y_pred = model.predict(X_test)

        res = evaluate(y_test, y_pred)
        res.update({"experiment": "Exp4", "seed": seed})
        results.append(res)

        predictions.append(pd.DataFrame({
            "y_true": le.inverse_transform(y_test),
            "y_pred": le.inverse_transform(y_pred),
            "experiment": "Exp4",
            "seed": seed
        }))

    df_res = pd.DataFrame(results)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(RESULT_PATH, index=False)

    df_pred = pd.concat(predictions)
    df_pred.to_csv(PRED_PATH, index=False)

    print("\n✅ DONE")
    print(df_res.groupby("experiment").mean())

    return df_res


if __name__ == "__main__":
    run_study()