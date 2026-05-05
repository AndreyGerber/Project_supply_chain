import numpy as np


def explain_local_prediction(model, X, y_true=None):
    """
    Approximates feature contribution via perturbation (fast & robust for XGBoost).
    """

    base_pred = model.predict_proba(X)[0]
    base_class = np.argmax(base_pred)

    importances = []

    X_np = X.copy().toarray()

    for i in range(X_np.shape[1]):

        X_perturbed = X_np.copy()
        X_perturbed[:, i] = 0  # feature removal

        proba = model.predict_proba(X_perturbed)[0]
        diff = base_pred[base_class] - proba[base_class]

        importances.append(diff)

    return np.array(importances)