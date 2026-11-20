import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from oulad_data import best_threshold, prepare


SEEDS = [20261001, 20261002, 20261003, 20261004, 20261005]
OUT = Path("results/supervisor_highlight_experiments")
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


class ModuleLogitBiasModel(nn.Module):
    """code_module enters only as a learned scalar added to the final logit."""

    def __init__(self, n_known_modules, context_dim):
        super().__init__()
        self.unk_idx = n_known_modules
        self.lstm = nn.LSTM(3, 24, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(48, 24), nn.Tanh(), nn.Linear(24, 1))
        self.context = nn.Sequential(
            nn.Linear(context_dim, 32), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, 24), nn.ReLU(),
        )
        self.bproj = nn.Linear(48, 24)
        self.gate = nn.Sequential(nn.Linear(24 + 24, 24), nn.Sigmoid())
        self.base_out = nn.Sequential(
            nn.Linear(24, 24), nn.ReLU(), nn.Dropout(0.15), nn.Linear(24, 1)
        )
        self.module_bias = nn.Embedding(
            n_known_modules + 1, 1, padding_idx=self.unk_idx
        )

    def forward(self, seq, context, mod):
        hs, _ = self.lstm(seq)
        weights = torch.softmax(self.attn(hs).squeeze(-1), dim=1)
        pooled = (hs * weights[:, :, None]).sum(dim=1)
        behavior = self.bproj(pooled)
        contextual = self.context(context)
        gate = self.gate(torch.cat([behavior, contextual], dim=1))
        fused = gate * behavior + (1.0 - gate) * contextual
        return self.base_out(fused).squeeze(1) + self.module_bias(mod).squeeze(1)


def split_data(data):
    train = data[data.code_presentation.str.startswith("2013")].copy()
    train_ids = set(train.id_student)
    val_raw = data[data.code_presentation.eq("2014B")].copy()
    val = val_raw[~val_raw.id_student.isin(train_ids)].copy()
    used_ids = train_ids | set(val_raw.id_student)
    test_raw = data[data.code_presentation.eq("2014J")].copy()
    test = test_raw[~test_raw.id_student.isin(used_ids)].copy()
    return train, val, test


def metric_set(y, p, threshold):
    z = p >= threshold
    return {
        "recall": recall_score(y, z),
        "f1": f1_score(y, z),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "threshold": float(threshold),
    }


def context_and_sequences(data, cat, num, seq_cols, train, val, test):
    context_num = [c for c in num if c not in seq_cols]
    context_cat = [c for c in cat if c != "code_module"]
    prep = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), context_cat),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), context_num),
    ])
    h_train = prep.fit_transform(train).astype("float32")
    h_val = prep.transform(val).astype("float32")
    h_test = prep.transform(test).astype("float32")

    def raw_seq(frame):
        values = np.stack([
            frame[[f"clicks_w{w}", f"active_days_w{w}", f"resources_w{w}"]].values
            for w in range(6)
        ], axis=1).astype("float32")
        values[:, :, 0] = np.log1p(values[:, :, 0])
        values[:, :, 2] = np.log1p(values[:, :, 2])
        return values

    s_train = raw_seq(train)
    mu = s_train.mean((0, 1), keepdims=True)
    sd = s_train.std((0, 1), keepdims=True) + 1e-6
    return (
        h_train, h_val, h_test,
        (s_train - mu) / sd,
        (raw_seq(val) - mu) / sd,
        (raw_seq(test) - mu) / sd,
    )


def loader(s, h, m, y, shuffle=False, seed=SEEDS[0]):
    ds = TensorDataset(
        torch.tensor(s), torch.tensor(h), torch.tensor(m),
        torch.tensor(y.astype("float32")),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=256, shuffle=shuffle, generator=generator)


def predict(net, s, h, m, y):
    net.eval()
    output = []
    with torch.no_grad():
        for sb, hb, mb, _ in loader(s, h, m, y):
            output.append(torch.sigmoid(net(sb, hb, mb)).numpy())
    return np.concatenate(output)


def run_logit_bias(data, cat, num, seq_cols, train, val, test):
    modules = sorted(train.code_module.unique())
    module_map = {module: index for index, module in enumerate(modules)}
    unknown = len(modules)

    def module_ids(frame):
        return np.array([module_map.get(v, unknown) for v in frame.code_module], dtype="int64")

    h_train, h_val, h_test, s_train, s_val, s_test = context_and_sequences(
        data, cat, num, seq_cols, train, val, test
    )
    known_mask = test.code_module.to_numpy() != "CCC"
    positive = train.risk.sum()
    negative = len(train) - positive
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative / positive], dtype=torch.float32)
    )
    rows = []
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        net = ModuleLogitBiasModel(len(modules), h_train.shape[1])
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        train_loader = loader(
            s_train, h_train, module_ids(train), train.risk.to_numpy(), True, seed
        )
        best = None
        patience = 0
        for epoch in range(1, 31):
            net.train()
            for sb, hb, mb, yb in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(net(sb, hb, mb), yb)
                loss.backward()
                optimizer.step()
            p_val = predict(net, s_val, h_val, module_ids(val), val.risk.to_numpy())
            score = average_precision_score(val.risk, p_val)
            if best is None or score > best[0] + 1e-5:
                best = (
                    score,
                    {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                    epoch,
                )
                patience = 0
            else:
                patience += 1
            if patience >= 6:
                break
        net.load_state_dict(best[1])
        p_val = predict(net, s_val, h_val, module_ids(val), val.risk.to_numpy())
        threshold = best_threshold(val.risk.to_numpy(), p_val)
        p_test = predict(net, s_test, h_test, module_ids(test), test.risk.to_numpy())
        for scope, mask in [
            ("all", np.ones(len(test), dtype=bool)),
            ("without_CCC", known_mask),
        ]:
            row = {
                "seed": seed,
                "variant": "module_logit_bias",
                "scope": scope,
                "best_epoch": best[2],
            }
            row.update(metric_set(test.risk.to_numpy()[mask], p_test[mask], threshold))
            rows.append(row)
        print(
            f"seed={seed} epoch={best[2]} all={rows[-2]['pr_auc']:.6f} "
            f"without_CCC={rows[-1]['pr_auc']:.6f}",
            flush=True,
        )
    return rows


def run_hgb_scopes(train, val, test, seq_cols):
    rows = []
    known_mask = test.code_module.to_numpy() != "CCC"
    for include_module in [True, False]:
        train_variant = train.copy()
        val_variant = val.copy()
        test_variant = test.copy()
        categorical = [
            "gender", "region", "highest_education", "imd_band", "age_band", "disability"
        ]
        if include_module:
            categorical = ["code_module"] + categorical
        numeric = ["num_of_prev_attempts", "studied_credits", "date_registration"] + seq_cols
        for frame in [train_variant, val_variant, test_variant]:
            for column in [c for c in seq_cols if c.startswith("clicks_") or c.startswith("resources_")]:
                frame[column] = np.log1p(frame[column])
        prep = ColumnTransformer([
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), numeric),
        ])
        x_train = prep.fit_transform(train_variant)
        x_val = prep.transform(val_variant)
        x_test = prep.transform(test_variant)
        model = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=24,
            l2_regularization=1.0, early_stopping=False, random_state=SEEDS[0],
        )
        model.fit(x_train, train_variant.risk)
        p_val = model.predict_proba(x_val)[:, 1]
        threshold = best_threshold(val_variant.risk.to_numpy(), p_val)
        p_test = model.predict_proba(x_test)[:, 1]
        for scope, mask in [
                ("all", np.ones(len(test_variant), dtype=bool)),
            ("without_CCC", known_mask),
        ]:
            row = {
                "variant": "hgb_with_module" if include_module else "hgb_without_module",
                "scope": scope,
            }
            row.update(metric_set(test_variant.risk.to_numpy()[mask], p_test[mask], threshold))
            rows.append(row)
    return rows


def main():
    data, cat, num, seq_cols = prepare()
    train, val, test = split_data(data)
    logit_rows = run_logit_bias(data, cat, num, seq_cols, train, val, test)
    pd.DataFrame(logit_rows).to_csv(OUT / "module_logit_bias_runs.csv", index=False)
    summary = (
        pd.DataFrame(logit_rows)
        .groupby(["scope", "variant"])[["recall", "f1", "roc_auc", "pr_auc", "brier", "best_epoch"]]
        .agg(["mean", "std"])
    )
    summary.to_csv(OUT / "module_logit_bias_summary.csv")

    # Reload to avoid applying log1p twice across the two HGB variants.
    data_hgb, _, _, seq_hgb = prepare()
    train_hgb, val_hgb, test_hgb = split_data(data_hgb)
    hgb_rows = run_hgb_scopes(train_hgb, val_hgb, test_hgb, seq_hgb)
    pd.DataFrame(hgb_rows).to_csv(OUT / "hgb_scopes.csv", index=False)

    existing = pd.read_csv("results/oulad_multiseed/runs.csv")
    neural = (
        existing.groupby(["scope", "variant"])["pr_auc"]
        .agg(["mean", "std"])
        .reset_index()
        .to_dict(orient="records")
    )
    result = {
        "seeds": SEEDS,
        "split": {"train": len(train), "validation": len(val), "test": len(test)},
        "neural_existing_pr_auc": neural,
        "module_logit_bias": json.loads(summary.reset_index().to_json(orient="records")),
        "hgb_scopes": hgb_rows,
    }
    (OUT / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string())
    print(pd.DataFrame(hgb_rows).to_string(index=False))


if __name__ == "__main__":
    main()
