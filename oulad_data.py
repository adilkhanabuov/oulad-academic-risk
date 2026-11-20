"""Shared OULAD preparation utilities for exact early-prediction windows."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


HORIZON_WEEKS = int(os.environ.get("OULAD_HORIZON_WEEKS", "6"))
CUTOFF = 7 * HORIZON_WEEKS
RAW = Path("data/oulad/raw")
OUT = Path(os.environ.get("OULAD_OUT", f"results/oulad_week{HORIZON_WEEKS}"))
OUT.mkdir(parents=True, exist_ok=True)


def build_weekly():
    """Aggregate VLE events in exact half-open windows [0, 7 * weeks)."""
    cache = OUT / "weekly_features_exact.csv"
    if cache.exists():
        return pd.read_csv(cache)
    parts = []
    columns = [
        "code_module", "code_presentation", "id_student",
        "id_site", "date", "sum_click",
    ]
    for chunk in pd.read_csv(
        RAW / "studentVle.csv", usecols=columns, chunksize=750_000
    ):
        chunk = chunk[(chunk.date >= 0) & (chunk.date < CUTOFF)]
        if chunk.empty:
            continue
        chunk["week"] = (chunk.date // 7).astype("int8")
        parts.append(chunk)
    if not parts:
        raise RuntimeError("No VLE events were found in the requested observation window.")
    filtered = pd.concat(parts, ignore_index=True)
    grouped = (
        filtered.groupby(
            ["code_module", "code_presentation", "id_student", "week"],
            observed=True,
        )
        .agg(
            clicks=("sum_click", "sum"),
            active_days=("date", "nunique"),
            resources=("id_site", "nunique"),
        )
        .reset_index()
    )
    keys = ["code_module", "code_presentation", "id_student"]
    wide = grouped.pivot_table(
        index=keys,
        columns="week",
        values=["clicks", "active_days", "resources"],
        fill_value=0,
    )
    wide.columns = [f"{kind}_w{week}" for kind, week in wide.columns]
    wide = wide.reset_index()
    for kind in ["clicks", "active_days", "resources"]:
        for week in range(HORIZON_WEEKS):
            column = f"{kind}_w{week}"
            if column not in wide:
                wide[column] = 0
    wide.to_csv(cache, index=False)
    return wide


def prepare():
    """Return the eligible cohort and features known at the forecast time."""
    info = pd.read_csv(RAW / "studentInfo.csv")
    registration = pd.read_csv(RAW / "studentRegistration.csv")
    registration["date_registration"] = pd.to_numeric(
        registration["date_registration"], errors="coerce"
    )
    registration["date_unregistration"] = pd.to_numeric(
        registration["date_unregistration"], errors="coerce"
    )
    keys = ["code_module", "code_presentation", "id_student"]
    data = info.merge(
        registration[keys + ["date_registration", "date_unregistration"]],
        on=keys,
        how="left",
    )
    data = data[
        data.date_unregistration.isna() | (data.date_unregistration >= CUTOFF)
    ].copy()
    data["risk"] = data.final_result.isin(["Fail", "Withdrawn"]).astype("int8")
    data = data.merge(build_weekly(), on=keys, how="left")
    sequence_columns = [
        f"{kind}_w{week}"
        for week in range(HORIZON_WEEKS)
        for kind in ["clicks", "active_days", "resources"]
    ]
    data[sequence_columns] = data[sequence_columns].fillna(0)
    categorical = [
        "code_module", "gender", "region", "highest_education",
        "imd_band", "age_band", "disability",
    ]
    numeric = [
        "num_of_prev_attempts", "studied_credits", "date_registration",
    ] + sequence_columns
    return data, categorical, numeric, sequence_columns


def expected_calibration_error(y, probability, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(
        np.digitize(probability, edges[1:-1], right=True), 0, n_bins - 1
    )
    value = 0.0
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if mask.any():
            value += mask.mean() * abs(
                float(probability[mask].mean()) - float(y[mask].mean())
            )
    return float(value)


def metrics(y, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction),
        "f1": f1_score(y, prediction),
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "brier": brier_score_loss(y, probability),
        "ece_10": expected_calibration_error(y, probability, 10),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }


def best_threshold(y, probability):
    candidates = np.linspace(0.10, 0.90, 161)
    return float(max(candidates, key=lambda t: f1_score(y, probability >= t)))

