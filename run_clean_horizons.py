import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
SEEDS = [20261001, 20261002, 20261003, 20261004, 20261005]


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    return float(sum((ids == b).mean() * abs(p[ids == b].mean() - y[ids == b].mean())
                     for b in range(bins) if (ids == b).any()))


def run_one(horizon):
    os.environ["OULAD_HORIZON_WEEKS"] = str(horizon)
    os.environ["OULAD_OUT"] = str(ROOT / f"results/oulad_week{horizon}")
    from oulad_data import prepare, best_threshold
    from run_oulad_multiseed import AblationModel, metric_set

    data, cat, num, seq_cols = prepare()
    train = data[data.code_presentation.str.startswith("2013")].copy()
    train_ids = set(train.id_student)
    val_raw = data[data.code_presentation.eq("2014B")].copy()
    val = val_raw[~val_raw.id_student.isin(train_ids)].copy()
    used_ids = train_ids | set(val_raw.id_student)
    test_raw = data[data.code_presentation.eq("2014J")].copy()
    test = test_raw[~test_raw.id_student.isin(used_ids)].copy()

    modules = sorted(train.code_module.unique())
    module_map = {m: i for i, m in enumerate(modules)}
    unk_idx = len(modules)
    def module_ids(frame):
        return np.array([module_map.get(v, unk_idx) for v in frame.code_module], dtype="int64")

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
        arr = np.stack([
            frame[[f"clicks_w{w}", f"active_days_w{w}", f"resources_w{w}"]].values
            for w in range(horizon)
        ], 1).astype("float32")
        arr[:, :, 0] = np.log1p(arr[:, :, 0])
        arr[:, :, 2] = np.log1p(arr[:, :, 2])
        return arr

    s_train = raw_seq(train)
    mu = s_train.mean((0, 1), keepdims=True)
    sd = s_train.std((0, 1), keepdims=True) + 1e-6
    s_train = (s_train - mu) / sd
    s_val = (raw_seq(val) - mu) / sd
    s_test = (raw_seq(test) - mu) / sd

    def loader(s, h, m, y, shuffle=False, seed=0):
        ds = TensorDataset(torch.tensor(s), torch.tensor(h), torch.tensor(m), torch.tensor(y.astype("float32")))
        gen = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=256, shuffle=shuffle, generator=gen)

    def predict(net, s, h, m, y):
        net.eval(); out = []
        with torch.no_grad():
            for sb, hb, mb, _ in loader(s, h, m, y):
                out.append(torch.sigmoid(net(sb, hb, mb)).numpy())
        return np.concatenate(out)

    pos = train.risk.sum(); neg = len(train) - pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
    rows = []
    module_rows = []
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        net = AblationModel(len(modules), h_train.shape[1], module_mode="full", use_attention=True, use_gate=True)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        train_loader = loader(s_train, h_train, module_ids(train), train.risk.values, True, seed)
        best = None; patience = 0
        for epoch in range(1, 31):
            net.train()
            for sb, hb, mb, yb in train_loader:
                opt.zero_grad(); loss = loss_fn(net(sb, hb, mb), yb); loss.backward(); opt.step()
            pv = predict(net, s_val, h_val, module_ids(val), val.risk.values)
            score = average_precision_score(val.risk, pv)
            if best is None or score > best[0] + 1e-5:
                best = (score, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, epoch)
                patience = 0
            else:
                patience += 1
            if patience >= 6: break
        net.load_state_dict(best[1])
        pv = predict(net, s_val, h_val, module_ids(val), val.risk.values)
        pt = predict(net, s_test, h_test, module_ids(test), test.risk.values)
        threshold = best_threshold(val.risk.values, pv)
        row = {"horizon": horizon, "seed": seed, "best_epoch": best[2], "threshold": threshold}
        row.update(metric_set(test.risk.values, pt, threshold))
        row["ece_10"] = ece(test.risk.values, pt, 10)
        rows.append(row)
        for module in sorted(test.code_module.unique()):
            mask = test.code_module.to_numpy() == module
            module_row = {
                "horizon": horizon,
                "seed": seed,
                "module": module,
                "n": int(mask.sum()),
                "risk_share": float(test.risk.to_numpy()[mask].mean()),
                "best_epoch": best[2],
                "threshold": threshold,
            }
            module_row.update(metric_set(test.risk.to_numpy()[mask], pt[mask], threshold))
            module_row["ece_10"] = ece(test.risk.to_numpy()[mask], pt[mask], 10)
            module_rows.append(module_row)
        print(horizon, seed, row, flush=True)

    return rows, module_rows, {
        "horizon": horizon,
        "registrations": {"train": len(train), "validation": len(val), "test": len(test)},
        "unique_students": {"train": train.id_student.nunique(), "validation": val.id_student.nunique(), "test": test.id_student.nunique()},
        "risk_share": {"train": float(train.risk.mean()), "validation": float(val.risk.mean()), "test": float(test.risk.mean())},
        "overlap": {
            "train_validation": len(set(train.id_student) & set(val.id_student)),
            "train_test": len(set(train.id_student) & set(test.id_student)),
            "validation_test": len(set(val.id_student) & set(test.id_student)),
        },
    }


def worker(horizon):
    rows, module_rows, meta = run_one(horizon)
    out = ROOT / f"results/oulad_clean_horizon_{horizon}"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "runs.csv", index=False)
    module_frame = pd.DataFrame(module_rows)
    module_frame.to_csv(out / "module_runs.csv", index=False)
    metrics = ["recall", "f1", "roc_auc", "pr_auc", "brier", "ece_10", "best_epoch"]
    summary = frame[metrics].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")
    module_metrics = ["recall", "f1", "roc_auc", "pr_auc", "brier", "ece_10"]
    module_summary = module_frame.groupby("module")[module_metrics].agg(["mean", "std"])
    module_summary.to_csv(out / "module_summary.csv")
    meta["summary"] = {m: {"mean": float(frame[m].mean()), "std": float(frame[m].std(ddof=1))} for m in metrics}
    meta["module_summary"] = {
        module: {
            metric: {
                "mean": float(group[metric].mean()),
                "std": float(group[metric].std(ddof=1)),
            }
            for metric in module_metrics
        }
        for module, group in module_frame.groupby("module")
    }
    (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        worker(int(sys.argv[1]))
    else:
        for h in (2, 4, 8):
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / ".deps/torch_runtime") + os.pathsep + env.get("PYTHONPATH", "")
            subprocess.run([sys.executable, __file__, str(h)], check=True, env=env, cwd=ROOT)
