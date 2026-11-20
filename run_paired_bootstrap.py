import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from oulad_data import prepare, best_threshold
from run_oulad_multiseed import AblationModel


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/oulad_paired_bootstrap"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [20261001, 20261002, 20261003, 20261004, 20261005]
N_BOOT = 2000
torch.set_num_threads(8)


def main():
    data, cat, num, seq_cols = prepare()
    train = data[data.code_presentation.str.startswith("2013")].copy()
    train_ids = set(train.id_student)
    val_raw = data[data.code_presentation.eq("2014B")].copy()
    val = val_raw[~val_raw.id_student.isin(train_ids)].copy()
    used_ids = train_ids | set(val_raw.id_student)
    test_raw = data[data.code_presentation.eq("2014J")].copy()
    test = test_raw[~test_raw.id_student.isin(used_ids)].copy()

    # Tabular baselines use the same feature set and train-fitted preprocessing.
    categorical = [
        "code_module", "gender", "region", "highest_education",
        "imd_band", "age_band", "disability",
    ]
    numeric = ["num_of_prev_attempts", "studied_credits", "date_registration"] + seq_cols
    tabular_prep = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), numeric),
    ])
    tab_train, tab_val, tab_test = train.copy(), val.copy(), test.copy()
    log_columns = [c for c in seq_cols if c.startswith("clicks_") or c.startswith("resources_")]
    for frame in (tab_train, tab_val, tab_test):
        frame[log_columns] = np.log1p(frame[log_columns])
    x_train = tabular_prep.fit_transform(tab_train)
    x_val = tabular_prep.transform(tab_val)
    x_test = tabular_prep.transform(tab_test)
    y_train = train.risk.to_numpy()
    y_val = val.risk.to_numpy()
    y_test = test.risk.to_numpy()

    lr = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEEDS[0])
    hgb = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.06, max_leaf_nodes=24,
        l2_regularization=1.0, early_stopping=False, random_state=SEEDS[0],
    )
    lr.fit(x_train, y_train)
    hgb.fit(x_train, y_train)
    lr_test = lr.predict_proba(x_test)[:, 1]
    hgb_test = hgb.predict_proba(x_test)[:, 1]

    # Inputs for the neural model, matching the five-seed ablation protocol.
    modules = sorted(train.code_module.unique())
    module_map = {module: index for index, module in enumerate(modules)}
    unknown_index = len(modules)

    def module_ids(frame):
        return np.array([module_map.get(v, unknown_index) for v in frame.code_module], dtype="int64")

    context_num = [c for c in num if c not in seq_cols]
    context_cat = [c for c in cat if c != "code_module"]
    context_prep = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), context_cat),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), context_num),
    ])
    h_train = context_prep.fit_transform(train).astype("float32")
    h_val = context_prep.transform(val).astype("float32")
    h_test = context_prep.transform(test).astype("float32")

    def raw_seq(frame):
        arr = np.stack([
            frame[[f"clicks_w{w}", f"active_days_w{w}", f"resources_w{w}"]].values
            for w in range(6)
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
        ds = TensorDataset(
            torch.tensor(s), torch.tensor(h), torch.tensor(m), torch.tensor(y.astype("float32"))
        )
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(ds, batch_size=256, shuffle=shuffle, generator=generator)

    def predict(net, s, h, m, y):
        net.eval()
        values = []
        with torch.no_grad():
            for sb, hb, mb, _ in loader(s, h, m, y):
                values.append(torch.sigmoid(net(sb, hb, mb)).numpy())
        return np.concatenate(values)

    pos = y_train.sum()
    neg = len(y_train) - pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
    neural_test = []
    best_epochs = []
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        net = AblationModel(
            len(modules), h_train.shape[1], module_mode="full", use_attention=True, use_gate=True
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        train_loader = loader(s_train, h_train, module_ids(train), y_train, True, seed)
        best = None
        patience = 0
        for epoch in range(1, 31):
            net.train()
            for sb, hb, mb, yb in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(net(sb, hb, mb), yb)
                loss.backward()
                optimizer.step()
            val_probability = predict(net, s_val, h_val, module_ids(val), y_val)
            score = average_precision_score(y_val, val_probability)
            if best is None or score > best[0] + 1e-5:
                best = (score, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, epoch)
                patience = 0
            else:
                patience += 1
            if patience >= 6:
                break
        net.load_state_dict(best[1])
        neural_test.append(predict(net, s_test, h_test, module_ids(test), y_test))
        best_epochs.append(best[2])
        print(seed, best[2], average_precision_score(y_test, neural_test[-1]), flush=True)

    neural_test = np.stack(neural_test)
    np.savez_compressed(
        OUT / "test_predictions.npz",
        y_test=y_test,
        id_student=test.id_student.to_numpy(),
        code_module=test.code_module.to_numpy(),
        neural=neural_test,
        logistic=lr_test,
        hgb=hgb_test,
        seeds=np.array(SEEDS),
    )

    observed_neural = np.mean([average_precision_score(y_test, p) for p in neural_test])
    observed = {
        "hybrid_mean_pr_auc": float(observed_neural),
        "logistic_pr_auc": float(average_precision_score(y_test, lr_test)),
        "hgb_pr_auc": float(average_precision_score(y_test, hgb_test)),
    }

    rng = np.random.default_rng(20261119)
    positive = np.flatnonzero(y_test == 1)
    negative = np.flatnonzero(y_test == 0)
    boot = {"logistic": [], "hgb": []}
    for _ in range(N_BOOT):
        ix = np.concatenate([
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ])
        selected_seeds = rng.integers(0, len(SEEDS), size=len(SEEDS))
        neural_ap = np.mean([
            average_precision_score(y_test[ix], neural_test[s, ix]) for s in selected_seeds
        ])
        boot["logistic"].append(neural_ap - average_precision_score(y_test[ix], lr_test[ix]))
        boot["hgb"].append(neural_ap - average_precision_score(y_test[ix], hgb_test[ix]))

    comparisons = {}
    for name, values in boot.items():
        values = np.asarray(values)
        baseline_key = "logistic_pr_auc" if name == "logistic" else "hgb_pr_auc"
        comparisons[name] = {
            "observed_delta": float(observed_neural - observed[baseline_key]),
            "hierarchical_stratified_bootstrap_95ci": [
                float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
            ],
            "probability_delta_gt_zero": float(np.mean(values > 0)),
        }

    result = {
        "seeds": SEEDS,
        "bootstrap_replicates": N_BOOT,
        "bootstrap_design": "stratified resampling of test registrations plus resampling of neural seeds",
        "n_test": len(test),
        "best_epochs": best_epochs,
        "observed": observed,
        "comparisons_hybrid_minus_baseline": comparisons,
    }
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
