import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from oulad_data import prepare, best_threshold


BASE_SEED = 20261001
SEEDS = [20261001, 20261002, 20261003, 20261004, 20261005]
OUT = Path("results/oulad_multiseed")
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


class AblationModel(nn.Module):
    def __init__(self, n_known_modules, context_dim, module_mode="full", use_attention=True, use_gate=True):
        super().__init__()
        self.module_mode = module_mode
        self.use_attention = use_attention
        self.use_gate = use_gate
        self.unk_idx = n_known_modules
        # padding_idx keeps the unseen-module vector exactly zero.
        self.module_emb = nn.Embedding(n_known_modules + 1, 8, padding_idx=self.unk_idx)
        self.lstm = nn.LSTM(3, 24, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(48 + 8, 24), nn.Tanh(), nn.Linear(24, 1))
        self.context = nn.Sequential(
            nn.Linear(context_dim, 32), nn.ReLU(), nn.Dropout(.15),
            nn.Linear(32, 24), nn.ReLU(),
        )
        self.bproj = nn.Linear(48, 24)
        self.gate = nn.Sequential(nn.Linear(24 + 24 + 8, 24), nn.Sigmoid())
        self.out = nn.Sequential(nn.Linear(24 + 8, 24), nn.ReLU(), nn.Dropout(.15), nn.Linear(24, 1))

    def forward(self, seq, context, mod):
        raw_e = self.module_emb(mod)
        if self.module_mode == "none":
            cond_e = torch.zeros_like(raw_e)
            out_e = torch.zeros_like(raw_e)
        elif self.module_mode == "output_only":
            cond_e = torch.zeros_like(raw_e)
            out_e = raw_e
        else:
            cond_e = raw_e
            out_e = raw_e
        hs, _ = self.lstm(seq)
        if self.use_attention:
            er = cond_e[:, None, :].expand(-1, hs.size(1), -1)
            weights = torch.softmax(self.attn(torch.cat([hs, er], 2)).squeeze(-1), 1)
            pooled = (hs * weights[:, :, None]).sum(1)
        else:
            pooled = hs.mean(1)
        behavior = self.bproj(pooled)
        contextual = self.context(context)
        if self.use_gate:
            gate = self.gate(torch.cat([behavior, contextual, cond_e], 1))
            fused = gate * behavior + (1 - gate) * contextual
        else:
            fused = 0.5 * behavior + 0.5 * contextual
        return self.out(torch.cat([fused, out_e], 1)).squeeze(1)


def metric_set(y, p, threshold):
    z = (p >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, z),
        "precision": precision_score(y, z, zero_division=0),
        "recall": recall_score(y, z),
        "f1": f1_score(y, z),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "threshold": float(threshold),
    }


def main():
    d, cat, num, seq_cols = prepare()
    train = d[d.code_presentation.str.startswith("2013")].copy()
    train_ids = set(train.id_student)
    val_raw = d[d.code_presentation.eq("2014B")].copy()
    val = val_raw[~val_raw.id_student.isin(train_ids)].copy()
    used_ids = train_ids | set(val_raw.id_student)
    test_raw = d[d.code_presentation.eq("2014J")].copy()
    test = test_raw[~test_raw.id_student.isin(used_ids)].copy()

    overlap = {
        "train_val": len(set(train.id_student) & set(val.id_student)),
        "train_test": len(set(train.id_student) & set(test.id_student)),
        "val_test": len(set(val.id_student) & set(test.id_student)),
    }
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
        a = np.stack([
            frame[[f"clicks_w{w}", f"active_days_w{w}", f"resources_w{w}"]].values
            for w in range(6)
        ], 1).astype("float32")
        a[:, :, 0] = np.log1p(a[:, :, 0]); a[:, :, 2] = np.log1p(a[:, :, 2])
        return a
    s_train = raw_seq(train)
    mu = s_train.mean((0, 1), keepdims=True)
    sd = s_train.std((0, 1), keepdims=True) + 1e-6
    s_train = (s_train - mu) / sd
    s_val = (raw_seq(val) - mu) / sd
    s_test = (raw_seq(test) - mu) / sd

    def loader(s, h, m, y, shuffle=False, seed=BASE_SEED):
        ds = TensorDataset(torch.tensor(s), torch.tensor(h), torch.tensor(m), torch.tensor(y.astype("float32")))
        gen = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=256, shuffle=shuffle, generator=gen)

    def predict(net, s, h, m, y):
        net.eval(); out = []
        with torch.no_grad():
            for sb, hb, mb, _ in loader(s, h, m, y):
                out.append(torch.sigmoid(net(sb, hb, mb)).numpy())
        return np.concatenate(out)

    variants = {
        "full_conditioned": dict(module_mode="full", use_attention=True, use_gate=True),
        "no_module": dict(module_mode="none", use_attention=True, use_gate=True),
        "module_output_only": dict(module_mode="output_only", use_attention=True, use_gate=True),
        "no_attention": dict(module_mode="full", use_attention=False, use_gate=True),
        "no_learned_gate": dict(module_mode="full", use_attention=True, use_gate=False),
    }
    results = []
    pos = train.risk.sum(); neg = len(train) - pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
    known_mask = (test.code_module.values != "CCC")

    for seed in SEEDS:
        for name, flags in variants.items():
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            net = AblationModel(len(modules), h_train.shape[1], **flags)
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
                if patience >= 6:
                    break
            net.load_state_dict(best[1])
            pv = predict(net, s_val, h_val, module_ids(val), val.risk.values)
            pt = predict(net, s_test, h_test, module_ids(test), test.risk.values)
            threshold = best_threshold(val.risk.values, pv)
            for scope, mask in [("all", np.ones(len(test), dtype=bool)), ("without_CCC", known_mask)]:
                row = {"seed": seed, "variant": name, "scope": scope, "best_epoch": best[2]}
                row.update(metric_set(test.risk.values[mask], pt[mask], threshold))
                results.append(row)
            print(seed, name, "epoch", best[2], "all", round(results[-2]["pr_auc"], 5), "noCCC", round(results[-1]["pr_auc"], 5), flush=True)

    frame = pd.DataFrame(results)
    frame.to_csv(OUT / "runs.csv", index=False)
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier", "best_epoch"]
    summary = frame.groupby(["scope", "variant"])[metrics].agg(["mean", "std"])
    summary.to_csv(OUT / "summary.csv")
    meta = {
        "seeds": SEEDS,
        "split": {
            "train": len(train), "validation": len(val), "test": len(test),
            "test_without_CCC": int(known_mask.sum()), "test_CCC": int((~known_mask).sum()),
        },
        "overlap_after_filtering": overlap,
        "known_train_modules": modules,
        "unknown_test_modules": sorted(set(test.code_module) - set(modules)),
        "unknown_handling": "dedicated zero embedding via padding_idx",
        "summary": json.loads(summary.reset_index().to_json(orient="records")),
    }
    (OUT / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string())
    print(json.dumps({k: v for k, v in meta.items() if k != "summary"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
