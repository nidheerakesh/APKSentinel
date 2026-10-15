from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

from config.settings import TrainConfig, GRAPHS_DIR, MODEL_DIR
from model.gat import MalwareGAT
from model.dataset import load_all_graphs, temporal_split


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def compute_class_weights(data_list: list) -> torch.Tensor:
    labels = [d.y.item() for d in data_list]
    counts = [labels.count(0), labels.count(1)]
    total = sum(counts)
    if 0 in counts:
        return torch.ones(2)
    weights = [total / (2.0 * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        probs = F.softmax(out, dim=1)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.y.view(-1).cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
    }
    try:
        metrics["auc_roc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["auc_roc"] = 0.0

    return metrics


def train(config: TrainConfig | None = None):
    config = config or TrainConfig()
    device = torch.device("cpu")

    print("Loading graphs...")
    all_data = load_all_graphs(GRAPHS_DIR)
    if not all_data:
        print("No graphs found. Run pipeline/process.py first.")
        return

    print(f"Loaded {len(all_data)} graphs")
    train_data, val_data, test_data = temporal_split(
        all_data, config.train_ratio, config.val_ratio
    )
    print(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size)
    test_loader = DataLoader(test_data, batch_size=config.batch_size)

    model = MalwareGAT(
        in_channels=25,
        hidden_channels=config.hidden_channels,
        num_heads=config.num_heads,
        num_classes=config.num_classes,
        dropout=config.dropout,
    ).to(device)

    weights = compute_class_weights(train_data).to(device)
    criterion = FocalLoss(weight=weights, gamma=2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-5
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = 0
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(epoch)

        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | LR: {lr:.6f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / "best_gat.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(MODEL_DIR / "best_gat.pt", weights_only=True))
    test_metrics = evaluate(model, test_loader, device)

    print("\n--- Test Results ---")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    import math
    clean_metrics = {k: (0.0 if math.isnan(v) else v) for k, v in test_metrics.items()}

    results_path = MODEL_DIR / "results.json"
    results_path.write_text(json.dumps({
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "best_val_f1": best_f1,
        "test_metrics": clean_metrics,
    }, indent=2))
    print(f"\nResults saved to {results_path}")

    return model, test_metrics


def main():
    train()


if __name__ == "__main__":
    main()
