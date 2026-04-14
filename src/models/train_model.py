import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.features.build_features import generate_tfidf, generate_embeddings
from src.data.load_data import load_processed_data

from src.evaluation.evaluate import evaluate_classification
from src.evaluation.validation import (
    cross_validate_model,
    check_class_balance
)
from src.evaluation.interpretability import show_feature_importance

from src.utils.experiment_tracking import log_experiment

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models/model.joblib"


def prepare_target(df: pd.DataFrame, mode: str = "classification") -> pd.Series:
    if mode == "classification":
        return (df["rating"] >= 4).astype(int)
    else:
        return df["rating"]


def get_model_and_params(model_type: str):
    if model_type == "xgboost":
        model = XGBClassifier(eval_metric="logloss")

        param_grid = {
            "model__n_estimators": [50, 100],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.05, 0.1],
        }

    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)

        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 20],
        }

    else:
        raise ValueError("Unknown model type")

    return model, param_grid


def train_model(
    model_type: str = "xgboost",
    mode: str = "classification",
    use_tuning: bool = True,
    feature_type: str = "tfidf" 
):
    df = load_processed_data()
    df = df.dropna(subset=["review_text_clean", "rating"])

    y = prepare_target(df, mode)

    # =========================================
    # FEATURE SWITCH
    # =========================================

    if feature_type == "tfidf":

        preprocessor, df = generate_tfidf(df)

        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.2, random_state=42
        )

        model, param_grid = get_model_and_params(model_type)

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

    elif feature_type == "embedding":

        X = generate_embeddings(df, version="v1")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model, param_grid = get_model_and_params(model_type)

        pipeline = model  # ❗ kein sklearn pipeline nötig

    else:
        raise ValueError("Unknown feature type")

    # =========================================
    # 🔍 TUNING
    # =========================================

    if use_tuning:
        print("\n🔍 Running Hyperparameter Tuning...")

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
        print("\n⚡ Training without tuning...")
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        best_params = model.get_params()

    # =========================================
    # 📊 EVALUATION
    # =========================================

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\nModel: {model_type} | Features: {feature_type}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    evaluate_classification(y_test, y_pred)
    check_class_balance(y)

    # ⚠️ Cross-val nur für TF-IDF sinnvoll
    if feature_type == "tfidf":
        cross_validate_model(best_model, df, y)

    # Interpretability (nur bei kompatiblen Modellen)
    try:
        show_feature_importance(best_model)
    except:
        print("Feature importance not available")

    log_experiment(
        model_name=model_type,
        metrics={"accuracy": round(acc, 4), "f1_score": round(f1, 4)},
        params=best_params,
        mode=mode,
        use_tuning=use_tuning
    )

    joblib.dump(best_model, MODEL_PATH)
    print(f"\n💾 Model saved to {MODEL_PATH}")

    return best_model



if __name__ == "__main__":
    train_model(model_type="xgboost", feature_type="embedding", use_tuning=True)