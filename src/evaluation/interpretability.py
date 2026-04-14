# src/evaluation/interpretability.py

import numpy as np
import shap


def show_feature_importance(pipeline):
    """
    Zeigt Feature Importance für tree-based Modelle
    """
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        print("\nTop Feature Importances:")
        print(importances[:20])
    else:
        print("Model does not support feature importance.")


def explain_with_shap(pipeline, X_sample):
    """
    SHAP Erklärung für Modellentscheidungen
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessing"]

    X_transformed = preprocessor.transform(X_sample)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_transformed)

    shap.plots.beeswarm(shap_values)