# compare_models_full.py

import pandas as pd
import numpy as np
import torch
import random
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns

from src.data.load_data import get_data
from src.features.build_features import (
    preprocess_dataframe,
    generate_tfidf,
    generate_embeddings
)
from src.utils.data_cleaning import clean_raw_data
from src.utils.experiment_tracking import log_experiment
from src.evaluation.reporting import generate_html_report


# ---------------- GLOBAL SEED ----------------
SEED = 42

def set_global_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seed(SEED)


# ---------------- MODEL CONFIG ----------------
MODEL_GRID = {
    "random_forest": {
        "model": RandomForestClassifier(random_state=SEED),
        "params": {
            "model__n_estimators": [50, 100],
            "model__max_depth": [5, 10],
        }
    },
    "xgboost": {
        "model": XGBClassifier(eval_metric="logloss", random_state=SEED),
        "params": {
            "model__n_estimators": [50, 100],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.05, 0.1]
        }
    }
}


# ---------------- TARGET ----------------
def prepare_target(df: pd.DataFrame):
    return (df["rating"] >= 4).astype(int)


# ---------------- TRAIN TF-IDF ----------------
def train_tfidf(df: pd.DataFrame, y: pd.Series, use_tuning=False):

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=SEED
    )

    preprocessor, X_train = generate_tfidf(X_train)
    X_test = X_test.copy()

    X_train_feat = preprocessor.fit_transform(X_train)
    X_test_feat = preprocessor.transform(X_test)

    return run_models(
        X_train_feat, X_test_feat, y_train, y_test,
        feature_type="tfidf",
        use_tuning=use_tuning
    )


# ---------------- TRAIN EMBEDDINGS ----------------
def train_embeddings(df: pd.DataFrame, y: pd.Series, use_tuning=False):

    # WICHTIG: Embeddings EINMAL erzeugen
    X_all = generate_embeddings(df)

    # Danach splitten
    X_train_feat, X_test_feat, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=SEED
    )

    return run_models(
        X_train_feat, X_test_feat, y_train, y_test,
        feature_type="embeddings",
        use_tuning=use_tuning
    )


# ---------------- MODEL LOOP ----------------
def run_models(X_train, X_test, y_train, y_test, feature_type, use_tuning):

    results = []
    trained_models = {}

    for model_name, cfg in MODEL_GRID.items():
        print(f"\n🔹 Training {model_name} ({feature_type})...")

        model = cfg["model"]
        param_grid = cfg["params"]

        pipeline = Pipeline([("model", model)])

        if use_tuning:
            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=3,
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=1
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            best_params = grid.best_params_
        else:
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = model.get_params()

        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        log_experiment(
            model_name=model_name,
            metrics={"accuracy": acc, "f1_score": f1},
            params={**best_params, "seed": SEED},
            mode="classification",
            use_tuning=use_tuning,
            feature_type=feature_type,        # 🔥 FIX
            data_version="v1"                 # 🔥 optional aber empfohlen
)

        results.append({
            "model": model_name,
            "feature_type": feature_type,
            "accuracy": acc,
            "f1": f1
        })

        trained_models[f"{model_name}_{feature_type}"] = best_model

    return pd.DataFrame(results), trained_models


# ---------------- VISUALIZATION ----------------
def plot_model_comparison(df_all: pd.DataFrame):

    plt.figure()
    sns.barplot(data=df_all, x="model", y="f1", hue="feature_type")
    plt.title("F1 Score Comparison")
    plt.ylim(0, 1)
    plt.show()

    plt.figure()
    sns.barplot(data=df_all, x="model", y="accuracy", hue="feature_type")
    plt.title("Accuracy Comparison")
    plt.ylim(0, 1)
    plt.show()


# ---------------- BEST MODEL ----------------
def select_and_save_best_model(df_all: pd.DataFrame, trained_models: dict):

    best_idx = df_all["f1"].idxmax()
    best_row = df_all.loc[best_idx]

    key = f"{best_row['model']}_{best_row['feature_type']}"
    best_model = trained_models[key]

    model_path = Path("models/best_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(best_model, model_path)

    print("\n🏆 Best Model:")
    print(best_row)
    print(f"💾 Saved to {model_path}")

    return best_model


# ---------------- MAIN ----------------
if __name__ == "__main__":

    print("\n🚀 Starting Model Comparison")

    df = get_data(use_processed=True)
    df = clean_raw_data(df)
    df = preprocess_dataframe(df)
    
    df = df.dropna(subset=["review_text_clean", "rating"])

    y = prepare_target(df)

    # 🔥 getrennte Pipelines (sauber!)
    df_tfidf, models_tfidf = train_tfidf(df, y, use_tuning=True)
    df_emb, models_emb = train_embeddings(df, y, use_tuning=True)

    df_all = pd.concat([df_tfidf, df_emb], ignore_index=True)
    all_models = {**models_tfidf, **models_emb}

    plot_model_comparison(df_all)

    best_model = select_and_save_best_model(df_all, all_models)

    generate_html_report(df_all)

    print("\n✅ Done") 