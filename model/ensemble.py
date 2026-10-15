from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from config.settings import GRAPHS_DIR, MODEL_DIR
from model.gat import MalwareGAT
from model.dataset import load_all_graphs, temporal_split


def extract_graph_features(data_list: list) -> np.ndarray:
    model = MalwareGAT()
    features = []
    for data in data_list:
        batch = torch.zeros(data.x.size(0), dtype=torch.long)
        gf = model._graph_features(data.x, data.edge_index, batch)
        base = gf.squeeze(0).numpy()
        extra = _extra_structural_features(data)
        features.append(np.concatenate([base, extra]))
    return np.array(features)


def _extra_structural_features(data) -> np.ndarray:
    x = data.x
    edge_index = data.edge_index
    n = x.size(0)
    num_edges = edge_index.size(1)

    in_deg = torch.zeros(n)
    out_deg = torch.zeros(n)
    if num_edges > 0:
        ones = torch.ones(num_edges)
        in_deg.scatter_add_(0, edge_index[1], ones)
        out_deg.scatter_add_(0, edge_index[0], ones)
    total_deg = in_deg + out_deg

    if n > 2:
        deg_np = total_deg.numpy()
        mean_d = deg_np.mean()
        std_d = deg_np.std() + 1e-8
        skew = ((deg_np - mean_d) ** 3).mean() / (std_d ** 3)
        kurt = ((deg_np - mean_d) ** 4).mean() / (std_d ** 4) - 3.0
    else:
        skew = 0.0
        kurt = 0.0

    isolated_ratio = (total_deg == 0).float().mean().item()
    leaf_ratio = (total_deg == 1).float().mean().item()

    if n > 1:
        threshold = total_deg.mean() + 2 * total_deg.std()
        hub_ratio = (total_deg > threshold).float().mean().item()
    else:
        hub_ratio = 0.0

    total_in = in_deg.sum().item()
    total_out = out_deg.sum().item()
    io_ratio = total_in / (total_out + 1e-8) if total_out > 0 else 0.0

    if num_edges > 0:
        self_loops = (edge_index[0] == edge_index[1]).float().sum().item()
        self_loop_ratio = self_loops / num_edges
    else:
        self_loop_ratio = 0.0

    node_types = x[:, :6].argmax(dim=1)
    sensitivity_by_type = np.zeros(6)
    for t in range(6):
        mask = (node_types == t)
        if mask.any() and x.size(1) > 6:
            sensitivity_by_type[t] = x[mask, 6].mean().item()

    if x.size(1) > 6:
        high_sens_ratio = (x[:, 6] > 0.5).float().mean().item()
    else:
        high_sens_ratio = 0.0

    perm_count = (node_types == 2).sum().item()
    api_count = (node_types == 3).sum().item()
    net_count = (node_types == 4).sum().item()
    crypto_count = (node_types == 5).sum().item()

    # Edge density by direction
    if n > 0:
        avg_in = in_deg.mean().item()
        avg_out = out_deg.mean().item()
        max_in = in_deg.max().item() if n > 0 else 0
        max_out = out_deg.max().item() if n > 0 else 0
    else:
        avg_in = avg_out = max_in = max_out = 0.0

    # Feature-based: mean of non-type features per node type
    method_feat_mean = 0.0
    api_feat_mean = 0.0
    if x.size(1) > 6:
        method_mask = (node_types == 0)
        if method_mask.any():
            method_feat_mean = x[method_mask, 6:].mean().item()
        api_mask = (node_types == 3)
        if api_mask.any():
            api_feat_mean = x[api_mask, 6:].mean().item()

    return np.array([
        skew, kurt,
        isolated_ratio, leaf_ratio, hub_ratio,
        io_ratio, self_loop_ratio,
        *sensitivity_by_type,
        high_sens_ratio,
        np.log1p(perm_count), np.log1p(api_count), np.log1p(net_count),
    ])


def get_gat_probs(model: MalwareGAT, data_list: list) -> np.ndarray:
    model.eval()
    probs = []
    for data in data_list:
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            p = F.softmax(out, dim=1)
            probs.append(p[0, 1].item())
    return np.array(probs)


def train_ensemble():
    print("Loading graphs...")
    all_data = load_all_graphs(GRAPHS_DIR)
    if not all_data:
        print("No graphs found.")
        return

    print(f"Loaded {len(all_data)} graphs")
    train_data, val_data, test_data = temporal_split(all_data, 0.7, 0.15)
    print(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    print("Extracting graph features...")
    X_train = extract_graph_features(train_data)
    X_val = extract_graph_features(val_data)
    X_test = extract_graph_features(test_data)
    y_train = np.array([d.y.item() for d in train_data])
    y_val = np.array([d.y.item() for d in val_data])
    y_test = np.array([d.y.item() for d in test_data])

    gat_model = MalwareGAT()
    gat_path = MODEL_DIR / "best_gat.pt"
    if gat_path.exists():
        gat_model.load_state_dict(torch.load(gat_path, weights_only=True))
    gat_model.eval()

    print("Getting GAT probabilities...")
    gat_train = get_gat_probs(gat_model, train_data)
    gat_val = get_gat_probs(gat_model, val_data)
    gat_test = get_gat_probs(gat_model, test_data)

    X_train_g = np.column_stack([X_train, gat_train])
    X_val_g = np.column_stack([X_val, gat_val])
    X_test_g = np.column_stack([X_test, gat_test])

    print(f"Feature dims: {X_train.shape[1]} base + 1 GAT = {X_train_g.shape[1]} total")

    from xgboost import XGBClassifier

    candidates = {
        "GBM": GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=4, max_features='sqrt',
            random_state=42,
        ),
        "XGB": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, gamma=0.1,
            random_state=42, verbosity=0,
        ),
        "RF": RandomForestClassifier(
            n_estimators=500, max_depth=10,
            min_samples_leaf=3, max_features='sqrt',
            random_state=42, n_jobs=-1,
        ),
    }

    best_name = None
    best_model = None
    best_f1 = 0
    best_uses_gat = False
    best_X_test = None

    for name, clf in candidates.items():
        # Try without GAT prob
        clf_no = type(clf)(**clf.get_params())
        clf_no.fit(X_train, y_train)
        f1_no = f1_score(y_val, clf_no.predict(X_val))

        # Try with GAT prob
        clf_gat = type(clf)(**clf.get_params())
        clf_gat.fit(X_train_g, y_train)
        f1_gat = f1_score(y_val, clf_gat.predict(X_val_g))

        if f1_gat >= f1_no:
            f1_best = f1_gat
            chosen = clf_gat
            uses_gat = True
            X_t = X_test_g
        else:
            f1_best = f1_no
            chosen = clf_no
            uses_gat = False
            X_t = X_test

        print(f"  {name}: Val F1 = {f1_no:.4f} (no GAT) / {f1_gat:.4f} (+GAT) → {f1_best:.4f}")

        if f1_best > best_f1:
            best_f1 = f1_best
            best_name = name
            best_model = chosen
            best_uses_gat = uses_gat
            best_X_test = X_t

    print(f"\nBest: {best_name} (Val F1: {best_f1:.4f}, uses_gat={best_uses_gat})")

    # Also try soft voting ensemble of top classifiers
    print("\nTraining voting ensemble...")
    X_fit = X_train_g
    X_val_fit = X_val_g
    X_test_fit = X_test_g

    voter = VotingClassifier(
        estimators=[
            ('xgb', XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                gamma=0.1, random_state=42, verbosity=0,
            )),
            ('gbm', GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=4, max_features='sqrt',
                random_state=42,
            )),
            ('rf', RandomForestClassifier(
                n_estimators=500, max_depth=10, min_samples_leaf=3,
                max_features='sqrt', random_state=42, n_jobs=-1,
            )),
        ],
        voting='soft',
    )
    voter.fit(X_fit, y_train)
    voter_f1 = f1_score(y_val, voter.predict(X_val_fit))
    print(f"  Voting ensemble Val F1: {voter_f1:.4f}")

    if voter_f1 > best_f1:
        print("  → Voting ensemble wins")
        best_model = voter
        best_f1 = voter_f1
        best_name = "Voting"
        best_uses_gat = True
        best_X_test = X_test_fit

    test_pred = best_model.predict(best_X_test)
    test_prob = best_model.predict_proba(best_X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, test_pred),
        "f1": f1_score(y_test, test_pred, zero_division=0),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall": recall_score(y_test, test_pred, zero_division=0),
    }
    try:
        metrics["auc_roc"] = roc_auc_score(y_test, test_prob)
    except ValueError:
        metrics["auc_roc"] = 0.0

    print(f"\n--- Best Ensemble ({best_name}) Test Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    gbm_path = MODEL_DIR / "ensemble_gbm.pkl"
    with open(gbm_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nModel saved to {gbm_path}")

    results_path = MODEL_DIR / "ensemble_results.json"
    results_path.write_text(json.dumps({
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "best_val_f1": float(best_f1),
        "best_classifier": best_name,
        "uses_gat_prob": best_uses_gat,
        "num_features": best_X_test.shape[1],
        "test_metrics": metrics,
    }, indent=2))
    print(f"Results saved to {results_path}")

    print("\n--- GAT-only comparison ---")
    gat_pred = (gat_test >= 0.5).astype(int)
    gat_metrics = {
        "accuracy": accuracy_score(y_test, gat_pred),
        "f1": f1_score(y_test, gat_pred, zero_division=0),
    }
    for k, v in gat_metrics.items():
        print(f"  {k}: {v:.4f}")

    return best_model, metrics


if __name__ == "__main__":
    train_ensemble()
