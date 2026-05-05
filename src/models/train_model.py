from xml.parsers.expat import model

import pandas as pd
from pathlib import Path
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

#from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from scipy.sparse import hstack, csr_matrix

from src.features.build_features import (
    get_tfidf_pipeline,
    generate_embeddings,
    get_structured_features,
    #generate_tfidf
)

from src.data.load_data import load_processed_data

from src.evaluation.evaluate import evaluate_classification
from src.evaluation.validation import (
    cross_validate_model,
    check_class_balance
)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import cohen_kappa_score

from src.utils.experiment_tracking import log_experiment

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models/model.joblib"


# =========================================
# 🎯 TARGET
# =========================================
def prepare_target(df: pd.DataFrame) -> pd.Series:
    return df["rating"].astype(int)

# =========================================
# ⚖️ ORDINAL WEIGHTS
# =========================================

def create_ordinal_weights(y: np.ndarray) -> np.ndarray:
    """
    Higher penalty for distant misclassifications.
    """
    class_counts = np.bincount(y)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y]

    # Ordinal smoothing (distance awareness)
    ordinal_boost = np.abs(y - np.median(y)) + 1

    return sample_weights * ordinal_boost

# =========================================
# 🤖 MODEL CONFIG
# =========================================
def get_model_and_params(model_type: str, num_classes: int):

    if model_type == "xgboost":
        model = XGBClassifier(
            eval_metric="mlogloss",
            random_state=42,
            num_class=num_classes,
            objective="multi:softprob"
        )

        param_grid = {
            "model__n_estimators": [100, 300],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.05, 0.1],
        }

    elif model_type == "xgb_hybrid":
        model = XGBClassifier(
            eval_metric="mlogloss",
            random_state=42,
            num_class=num_classes,
            objective="multi:softprob"
        )

        param_grid = {
            "n_estimators": [300],
            "max_depth": [6],
            "learning_rate": [0.1],
        }

    else:
        raise ValueError("Unknown model type")

    return model, param_grid


# =========================================
# 🚀 TRAINING
# =========================================
def train_model(
    model_type: str = "xgb_hybrid",
    use_tuning: bool = True,
    feature_type: str = "hybrid"
):

    df = load_processed_data()
    df = df.dropna(subset=["review_text_clean_en", "rating"])

    y = prepare_target(df)

    le = LabelEncoder()
    y = le.fit_transform(y)
    num_classes = len(set(y))

    # =========================================
    # HYBRID FEATURE PIPELINE (FIXED)
    # =========================================

    if feature_type == "hybrid":

        X_text = df["review_text_clean_en"]

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text, y, test_size=0.2, random_state=42, stratify=y
        )

        # ---- TF-IDF ----
        tfidf_pipeline = get_tfidf_pipeline()
        X_train_tfidf = tfidf_pipeline.fit_transform(X_train_text)
        X_test_tfidf = tfidf_pipeline.transform(X_test_text)

        # ---- Align DataFrames ----
        df_train = df.loc[X_train_text.index]
        df_test = df.loc[X_test_text.index]

        # ---- Embeddings ----
        X_train_emb = generate_embeddings(df_train, version="v1")
        X_test_emb = generate_embeddings(df_test, version="v1")

        # ---- Structured ----
        X_train_struct = get_structured_features(df_train)
        X_test_struct = get_structured_features(df_test)

        # ---- Combine ----
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
        print(X_train_tfidf.shape)
        print(X_train_emb.shape)
        print(X_train_struct.shape)

        model, param_grid = get_model_and_params("xgb_hybrid", num_classes)
        use_pipeline = False

    else:
        raise ValueError("Only hybrid supported in this corrected version")

    # =========================================
    # 🔍 TUNING
    # =========================================

    if use_tuning:

        grid = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=1
        )

        #sample_weight = create_ordinal_weights(y_train)

        grid.fit(
            X_train,
            y_train,
            #sample_weight=sample_weight
        )
        best_model = grid.best_estimator_
        best_params = grid.best_params_

    else:
        #sample_weight = create_ordinal_weights(y_train)

        grid.fit(
            X_train,
            y_train,
            #sample_weight=sample_weight
        )
        best_model = model
        best_params = model.get_params()

    # =========================================
    # 📊 EVALUATION
    # =========================================

    y_pred = best_model.predict(X_test)

    class_dist = dict(pd.Series(y_train).value_counts(normalize=True))
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    mae = mean_absolute_error(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    
    print(f"\nModel: {model_type} | Features: {feature_type}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Kappa: {kappa:.4f}")

    evaluate_classification(y_test, y_pred)
    check_class_balance(y)

    if feature_type == "tfidf":
        cross_validate_model(best_model, df, y)

    #try:
    #    show_feature_importance(best_model)
    #except Exception:
    #    print("Feature importance not available")

    # =========================================
    # 📦 TRACKING
    # =========================================


    log_experiment(
    model_name=model_type,
    metrics={
        "accuracy": acc,
        "f1_score": f1,
        "mae": mae,
        "kappa": kappa
    },
    params=best_params,
    feature_type=feature_type,
    mode="MultiClassification",
    use_tuning=use_tuning,
    sampling_strategy="none",
    class_distribution_before=class_dist,
    class_distribution_after=class_dist,
    experiment_variant="ordinal_weighted"

)

    # =========================================
    # 💾 SAVE
    # =========================================

    joblib.dump({
        "model": best_model,
        "tfidf_pipeline": tfidf_pipeline,
        "label_encoder": le,
        "embedding_version": "v1"
    }, MODEL_PATH)

    print(f"\n💾 Model saved to {MODEL_PATH}")

    return best_model

if __name__ == "__main__":
    train_model(model_type="xgb_hybrid", feature_type="hybrid", use_tuning=True)