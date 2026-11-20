import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


SOURCE = Path("results/oulad_paired_bootstrap/test_predictions.npz")
OUT = Path("results/oulad_student_cluster_bootstrap")
N_BOOT = 2000
RANDOM_SEED = 20261120


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(SOURCE, allow_pickle=True)
    y = data["y_test"]
    student = data["id_student"]
    neural = data["neural"]
    logistic = data["logistic"]
    hgb = data["hgb"]
    seeds = data["seeds"]

    unique_students, inverse = np.unique(student, return_inverse=True)
    cluster_indices = [np.flatnonzero(inverse == i) for i in range(len(unique_students))]
    observed_neural = np.mean([average_precision_score(y, p) for p in neural])
    observed = {
        "hybrid_mean_pr_auc": float(observed_neural),
        "logistic_pr_auc": float(average_precision_score(y, logistic)),
        "hgb_pr_auc": float(average_precision_score(y, hgb)),
    }

    rng = np.random.default_rng(RANDOM_SEED)
    differences = {"logistic": [], "hgb": []}
    for _ in range(N_BOOT):
        sampled_clusters = rng.integers(0, len(unique_students), size=len(unique_students))
        index = np.concatenate([cluster_indices[i] for i in sampled_clusters])
        sampled_seeds = rng.integers(0, neural.shape[0], size=neural.shape[0])
        neural_ap = np.mean([
            average_precision_score(y[index], neural[s, index]) for s in sampled_seeds
        ])
        differences["logistic"].append(
            neural_ap - average_precision_score(y[index], logistic[index])
        )
        differences["hgb"].append(
            neural_ap - average_precision_score(y[index], hgb[index])
        )

    comparisons = {}
    for name, values in differences.items():
        values = np.asarray(values)
        baseline_key = "logistic_pr_auc" if name == "logistic" else "hgb_pr_auc"
        comparisons[name] = {
            "observed_delta": float(observed_neural - observed[baseline_key]),
            "student_cluster_bootstrap_95ci": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "probability_delta_gt_zero": float(np.mean(values > 0)),
        }

    result = {
        "bootstrap_replicates": N_BOOT,
        "bootstrap_design": (
            "student-cluster resampling of the 2014J test set plus resampling of five neural seeds"
        ),
        "n_registrations": int(len(y)),
        "n_student_clusters": int(len(unique_students)),
        "neural_seeds": seeds.tolist(),
        "observed": observed,
        "comparisons_hybrid_minus_baseline": comparisons,
    }
    (OUT / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
