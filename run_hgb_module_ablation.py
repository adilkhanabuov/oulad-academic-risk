import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/oulad/raw"
WEEKLY = ROOT / "results/oulad_week6/weekly_features_exact.csv"
OUT = ROOT / "results/oulad_hgb_module_ablation"
SEEDS = [20261001, 20261002, 20261003, 20261004, 20261005]
CUTOFF = 42


def load_data():
    info = pd.read_csv(RAW / "studentInfo.csv")
    reg = pd.read_csv(RAW / "studentRegistration.csv")
    reg["date_registration"] = pd.to_numeric(reg["date_registration"], errors="coerce")
    reg["date_unregistration"] = pd.to_numeric(reg["date_unregistration"], errors="coerce")
    keys = ["code_module", "code_presentation", "id_student"]
    data = info.merge(
        reg[keys + ["date_registration", "date_unregistration"]],
        on=keys,
        how="left",
    )
    data = data[
        data.date_unregistration.isna() | (data.date_unregistration >= CUTOFF)
    ].copy()
    data["risk"] = data.final_result.isin(["Fail", "Withdrawn"]).astype("int8")
    weekly = pd.read_csv(WEEKLY)
    data = data.merge(weekly, on=keys, how="left")
    seq = [
        f"{kind}_w{week}"
        for week in range(6)
        for kind in ["clicks", "active_days", "resources"]
    ]
    data[seq] = data[seq].fillna(0)
    for col in [c for c in seq if c.startswith("clicks_") or c.startswith("resources_")]:
        data[col] = np.log1p(data[col])
    return data, seq


def split_without_student_overlap(data):
    train = data[data.code_presentation.str.startswith("2013")].copy()
    train_ids = set(train.id_student)
    validation_raw = data[data.code_presentation.eq("2014B")].copy()
    validation = validation_raw[~validation_raw.id_student.isin(train_ids)].copy()
    used_ids = train_ids | set(validation_raw.id_student)
    test_raw = data[data.code_presentation.eq("2014J")].copy()
    test = test_raw[~test_raw.id_student.isin(used_ids)].copy()
    return train, validation, test


def best_threshold(y_true, probability):
    candidates = np.linspace(0.10, 0.90, 161)
    return float(max(candidates, key=lambda t: f1_score(y_true, probability >= t)))


def run_variant(train, validation, test, seq, include_module, seed):
    categorical = [
        "gender", "region", "highest_education", "imd_band", "age_band", "disability"
    ]
    if include_module:
        categorical = ["code_module"] + categorical
    numeric = ["num_of_prev_attempts", "studied_credits", "date_registration"] + seq
    prep = ColumnTransformer(
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
    x_train = prep.fit_transform(train)
    x_validation = prep.transform(validation)
    x_test = prep.transform(test)
    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=24,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(x_train, train.risk)
    val_probability = model.predict_proba(x_validation)[:, 1]
    threshold = best_threshold(validation.risk.to_numpy(), val_probability)
    probability = model.predict_proba(x_test)[:, 1]
    prediction = probability >= threshold
    y = test.risk.to_numpy()
    return {
        "seed": seed,
        "variant": "with_code_module" if include_module else "without_code_module",
        "threshold": threshold,
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction),
        "f1": f1_score(y, prediction),
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "brier": brier_score_loss(y, probability),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, seq = load_data()
    train, validation, test = split_without_student_overlap(data)
    rows = []
    for seed in SEEDS:
        for include_module in [True, False]:
            result = run_variant(train, validation, test, seq, include_module, seed)
            rows.append(result)
            print(result, flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "runs.csv", index=False)
    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "threshold"]
    summary = runs.groupby("variant")[metrics].agg(["mean", "std"])
    summary.to_csv(OUT / "summary.csv")
    paired = runs.pivot(index="seed", columns="variant", values=metrics)
    deltas = {}
    for metric in metrics:
        values = paired[(metric, "with_code_module")] - paired[(metric, "without_code_module")]
        deltas[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    metadata = {
        "seeds": SEEDS,
        "split_registrations": {
            "train": len(train), "validation": len(validation), "test": len(test)
        },
        "split_unique_students": {
            "train": train.id_student.nunique(),
            "validation": validation.id_student.nunique(),
            "test": test.id_student.nunique(),
        },
        "student_overlap": {
            "train_validation": len(set(train.id_student) & set(validation.id_student)),
            "train_test": len(set(train.id_student) & set(test.id_student)),
            "validation_test": len(set(validation.id_student) & set(test.id_student)),
        },
        "paired_delta_with_minus_without": deltas,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
