import json
import hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd

TRACKING_PATH = Path("models/runs.json")
    
    
# ---------------- RUN ID ----------------
def generate_run_id(model_name: str, params: dict, metrics: dict, feature_type: str, sampling_strategy: str, class_distribution_before: dict, class_distribution_after: dict, data_version: str) -> str:
    raw = json.dumps({
        "model": model_name,
        "params": params,
        "metrics": metrics,
        "feature_type": feature_type,
        "sampling_strategy": sampling_strategy,
        "class_distribution_before": class_distribution_before,
        "class_distribution_after": class_distribution_after,
        "data_version": data_version
    }, sort_keys=True, default=str)

    return hashlib.md5(raw.encode()).hexdigest()[:10]


# ---------------- JSON SAFETY ----------------
def make_json_serializable(obj):
    try:
        json.dumps(obj)
        return obj
    except:
        return str(obj)


def clean_dict(d: dict) -> dict:
    return {k: make_json_serializable(v) for k, v in d.items()}


# ---------------- LOAD RUNS ----------------
def load_runs() -> pd.DataFrame:
    if not TRACKING_PATH.exists():
        return pd.DataFrame()

    rows = []
    with open(TRACKING_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return pd.DataFrame(rows)


# ---------------- DEDUP CHECK ----------------
def run_exists(run_id: str) -> bool:
    if not TRACKING_PATH.exists():
        return False

    with open(TRACKING_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if run_id in line:
                return True
    return False


# ---------------- LOGGING ----------------
def log_experiment(
    model_name: str,
    metrics: dict,
    params: dict,
    mode: str ="unknown",
    use_tuning: bool =False,
    feature_type: str= "unknown",
    sampling_strategy: str =None,
    class_distribution_before: dict ={},
    class_distribution_after: dict ={},
    experiment_variant= "unknown",
    data_version: str = "v1"
):
    metrics_clean = clean_dict(metrics)
    params_clean = clean_dict(params)
    class_distribution_before_clean = clean_dict(class_distribution_before)
    class_distribution_after_clean = clean_dict(class_distribution_after)
    sampling_strategy_clean = make_json_serializable(sampling_strategy)

    run_id = generate_run_id(
        model_name,
        params_clean,
        metrics_clean,
        feature_type,
        sampling_strategy_clean,
        class_distribution_before_clean,
        class_distribution_after_clean,
        experiment_variant,
        data_version
    )

    if run_exists(run_id):
        print(f"⚠️ Run already exists (ID: {run_id}) → skipping")
        return

    run = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "feature_type": feature_type,
        "data_version": data_version,
        "metrics": metrics_clean,
        "params": params_clean,
        "mode": mode,
        "use_tuning": use_tuning,
        "sampling_strategy": sampling_strategy,
        "class_distribution_before": class_distribution_before,
        "class_distribution_after": class_distribution_after,
        "experiment_variant": experiment_variant,
        "data_version": data_version
    }

    TRACKING_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(TRACKING_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(run) + "\n")

    print(f"✅ Experiment logged (ID: {run_id})")


# ================================
# 🚀 ANALYSIS FUNCTIONS
# ================================

def get_best_run(metric: str = "f1_score"):
    df = load_runs()
    if df.empty:
        print("No runs found.")
        return None

    df["score"] = df["metrics"].apply(lambda x: x.get(metric, 0))
    best = df.sort_values(by="score", ascending=False).iloc[0]

    print("\n🏆 Best Run:")
    print(best)
    return best


def compare_feature_types(metric: str = "f1_score"):
    df = load_runs()
    if df.empty:
        print("No runs found.")
        return

    df["score"] = df["metrics"].apply(lambda x: x.get(metric, 0))

    grouped = df.groupby("feature_type")["score"].mean()

    print("\n📊 Average Performance by Feature Type:")
    print(grouped.sort_values(ascending=False))


def get_top_n_runs(n: int = 5, metric: str = "f1_score"):
    df = load_runs()
    if df.empty:
        print("No runs found.")
        return df

    df["score"] = df["metrics"].apply(lambda x: x.get(metric, 0))

    top = df.sort_values(by="score", ascending=False).head(n)

    print(f"\n🔝 Top {n} Runs:")
    print(top[["model", "feature_type", "score"]])

    return top


# ---------------- MAIN TEST ----------------
if __name__ == "__main__":
    print("\n=== Experiment Analysis ===")

    get_best_run()
    compare_feature_types()
    get_top_n_runs()