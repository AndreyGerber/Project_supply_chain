import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


# =========================================
# 🔧 PERMUTE BLOCK (SPARSE SAFE)
# =========================================
def permute_block_sparse(X, start, end):
    """
    Permutes a block of columns in a sparse matrix.
    """
    X_copy = X.copy().tocsr()

    for col in range(start, end):
        col_data = X_copy[:, col].toarray().ravel()
        np.random.shuffle(col_data)
        X_copy[:, col] = col_data.reshape(-1, 1)

    return X_copy


# =========================================
# 📊 MAIN ANALYSIS
# =========================================
def run_group_permutation_importance(
    model,
    X_test,
    y_test,
    tfidf_dim=5000,
    emb_dim=384,
    struct_dim=4,
    n_runs=5
):
    """
    Group-wise permutation importance for hybrid NLP models.
    """

    def score(y_true, y_pred):
        return f1_score(y_true, y_pred, average="weighted")

    # --- Baseline ---
    baseline_preds = model.predict(X_test)
    baseline_score = score(y_test, baseline_preds)

    print(f"\nBaseline F1 (weighted): {baseline_score:.4f}")

    results = {
        "TF-IDF": [],
        "Embeddings": [],
        "Structure": []
    }

    for run in range(n_runs):
        print(f"\nRun {run + 1}/{n_runs}")

        # --- TF-IDF ---
        X_perm = permute_block_sparse(X_test, 0, tfidf_dim)
        score_perm = score(y_test, model.predict(X_perm))
        results["TF-IDF"].append(baseline_score - score_perm)

        # --- Embeddings ---
        start = tfidf_dim
        end = tfidf_dim + emb_dim
        X_perm = permute_block_sparse(X_test, start, end)
        score_perm = score(y_test, model.predict(X_perm))
        results["Embeddings"].append(baseline_score - score_perm)

        # --- Structure ---
        start = tfidf_dim + emb_dim
        end = start + struct_dim
        X_perm = permute_block_sparse(X_test, start, end)
        score_perm = score(y_test, model.predict(X_perm))
        results["Structure"].append(baseline_score - score_perm)

    # --- Aggregation ---
    final_results = {
        k: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v))
        }
        for k, v in results.items()
    }

    print("\n=== Feature Group Importance ===")
    for k, v in final_results.items():
        print(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")

    return final_results


# =========================================
# 📈 PLOT
# =========================================
def plot_group_importance(results):
    groups = list(results.keys())
    means = [results[g]["mean"] for g in groups]
    stds = [results[g]["std"] for g in groups]

    plt.figure()
    plt.bar(groups, means, yerr=stds)
    plt.title("Feature Group Importance (Permutation)")
    plt.ylabel("Drop in F1 Score")
    plt.xlabel("Feature Groups")
    plt.tight_layout()
    plt.show()