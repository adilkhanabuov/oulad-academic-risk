from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_hgb_module_ablation import load_data, split_without_student_overlap


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/oulad_clean_baselines"
SEEDS = [20261001, 20261002, 20261003, 20261004, 20261005]


def best_threshold(y_true, probability):
    values = np.linspace(0.10, 0.90, 161)
    return float(max(values, key=lambda t: f1_score(y_true, probability >= t)))


def metric_set(y, probability, threshold):
    prediction = probability >= threshold
    return {
        "accuracy": accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction),
        "f1": f1_score(y, prediction),
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "brier": brier_score_loss(y, probability),
        "threshold": threshold,
    }


def make_preprocessor(categorical, numeric):
    return ColumnTransformer(
        [
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, seq = load_data()
    train, validation, test = split_without_student_overlap(data)
    categorical = [
        "code_module", "gender", "region", "highest_education",
        "imd_band", "age_band", "disability",
    ]
    numeric = ["num_of_prev_attempts", "studied_credits", "date_registration"] + seq
    prep = make_preprocessor(categorical, numeric)
    x_train = prep.fit_transform(train)
    x_validation = prep.transform(validation)
    x_test = prep.transform(test)
    y_train = train.risk.to_numpy()
    y_validation = validation.risk.to_numpy()
    y_test = test.risk.to_numpy()

    rows = []
    for seed in SEEDS:
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1500, class_weight="balanced", random_state=seed
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=350,
                min_samples_leaf=6,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.06,
                max_leaf_nodes=24,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=seed,
            ),
        }
        for name, model in models.items():
            model.fit(x_train, y_train)
            val_probability = model.predict_proba(x_validation)[:, 1]
            threshold = best_threshold(y_validation, val_probability)
            test_probability = model.predict_proba(x_test)[:, 1]
            row = {"seed": seed, "model": name}
            row.update(metric_set(y_test, test_probability, threshold))
            rows.append(row)
            print(row, flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "runs.csv", index=False)
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "threshold"]
    summary = runs.groupby("model")[metrics].agg(["mean", "std"])
    summary.to_csv(OUT / "summary.csv")

    # Add the already completed five-seed hybrid result from the same clean split.
    neural = pd.read_csv(ROOT / "results/oulad_multiseed/runs.csv")
    neural = neural[(neural.scope == "all") & (neural.variant == "full_conditioned")].copy()
    neural["model"] = "Hybrid BiLSTM"
    neural = neural[["seed", "model"] + metrics]
    combined = pd.concat([runs, neural], ignore_index=True)
    combined.to_csv(OUT / "combined_runs.csv", index=False)
    combined_summary = combined.groupby("model")[metrics].agg(["mean", "std"])
    combined_summary.to_csv(OUT / "combined_summary.csv")
    print("\nCOMBINED\n", combined_summary.to_string())


if __name__ == "__main__":
    main()
