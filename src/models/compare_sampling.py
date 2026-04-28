# compare_sampling.py

import numpy as np
import pandas as pd
import random

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error

from xgboost import XGBRegressor
from imblearn.over_sampling import SMOTE

from src.data.load_data import load_raw_data
from src.utils.data_cleaning import clean_raw_data
from src.features.build_features import preprocess_dataframe, generate_embeddings
from src.features.store_feature import FeatureStore
from src.features.build_features import generate_tfidf, generate_embeddings
from src.data.load_data import load_processed_data
from src.utils.experiment_tracking import log_experiment


# ================================
# 🔒 REPRODUCIBILITY
# ================================

SEEDS = [42, 1337, 2024]


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


# ================================
# 📊 METRICS
# ================================

def evaluate_all(y_true, y_pred_reg):
    y_pred_cls = np.clip(np.round(y_pred_reg), 1, 5)

    acc = accuracy_score(y_true, y_pred_cls)
    f1 = f1_score(y_true, y_pred_cls, average="macro")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_reg))
    cm = confusion_matrix(y_true, y_pred_cls)

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "rmse": rmse,
        "confusion_matrix": cm.tolist()
    }


# ================================
# 📉 SAMPLING STRATEGIEN
# ================================

def class_distribution(y):
    return dict(Counter(y))


def undersample_majority(X, y, max_ratio=3):
    df = pd.DataFrame(X)
    df["target"] = y

    counts = df["target"].value_counts()
    min_count = counts.min()

    dfs = []
    for cls, count in counts.items():
        df_cls = df[df["target"] == cls]

        if count > min_count * max_ratio:
            df_cls = df_cls.sample(min_count * max_ratio, random_state=42)

        dfs.append(df_cls)

    df_new = pd.concat(dfs).sample(frac=1, random_state=42)

    y_new = df_new["target"].values
    X_new = df_new.drop(columns=["target"]).values

    return X_new, y_new


def compute_class_weights(y):
    counts = Counter(y)
    total = sum(counts.values())

    weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
    return np.array([weights[val] for val in y])


def regression_focus_weights(y):
    return 1 + (5 - y)


def apply_smote(X, y):
    smote = SMOTE(sampling_strategy="not majority", random_state=42)
    return smote.fit_resample(X, y)


# ================================
# 🧠 MODEL TRAINING
# ================================

def train_model(X_train, y_train, sample_weight=None):
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist"
    )

    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


# ================================
# 🧪 EXPERIMENT RUN
# ================================

def run_experiment(name, X, y, strategy_fn):

    all_metrics = []

    for seed in SEEDS:
        set_seed(seed)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )

        before_dist = class_distribution(y_train)

        X_train_s, y_train_s, weights = strategy_fn(X_train, y_train)

        after_dist = class_distribution(y_train_s)

        model = train_model(X_train_s, y_train_s, weights)

        preds = model.predict(X_test)

        metrics = evaluate_all(y_test, preds)
        all_metrics.append(metrics)

        log_experiment(
            model_name="xgb_regressor",
            metrics=metrics,
            params=model.get_params(),
            mode="regression",
            use_tuning=False,
            feature_type="embeddings+structured",
            sampling_strategy=name,
            class_distribution_before=before_dist,
            class_distribution_after=after_dist
        )

    # Mittelwert berechnen
    avg_metrics = {
        k: np.mean([m[k] for m in all_metrics]) if k != "confusion_matrix" else all_metrics[0][k]
        for k in all_metrics[0]
    }

    return avg_metrics


# ================================
# 🎯 STRATEGIEN
# ================================

def baseline(X, y):
    return X, y, None


def undersample_plus_weights(X, y):
    X_new, y_new = undersample_majority(X, y)
    weights = compute_class_weights(y_new)
    return X_new, y_new, weights


def weights_only(X, y):
    return X, y, compute_class_weights(y)


def weights_plus_focus(X, y):
    w1 = compute_class_weights(y)
    w2 = regression_focus_weights(y)
    return X, y, w1 * w2


def smote_only(X, y):
    X_new, y_new = apply_smote(X, y)
    return X_new, y_new, None


# ================================
# 🚀 MAIN PIPELINE
# ================================

def main():
    print("🚀 Loading Data...")
    
    df = load_processed_data()
    df = df.dropna(subset=["review_text_clean", "rating"])

    # Features laden (Embeddings)
    X = generate_embeddings(df, version="S1", use_clean_text=False)

    y = df["rating"].values

    print("📊 Running Experiments...")

    results = []

    experiments = {
        "baseline": baseline,
        "undersample+weights": undersample_plus_weights,
        "weights_only": weights_only,
        "weights+focus": weights_plus_focus,
        "smote": smote_only
    }

    for name, fn in experiments.items():
        print(f"\n=== {name} ===")
        metrics = run_experiment(name, X, y, fn)
        metrics["experiment"] = name
        results.append(metrics)

    df_results = pd.DataFrame(results)
    df_results.to_csv("sampling_comparison.csv", index=False)

    print("\n✅ Results saved to sampling_comparison.csv")


if __name__ == "__main__":
    main()