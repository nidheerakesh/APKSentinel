from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

from model.gat import MalwareGAT
from config.settings import ExplainConfig


@dataclass
class ExplanationResult:
    node_importances: dict[str, float]
    top_nodes: list[tuple[str, float]]
    edge_importances: list[tuple[str, str, float]]
    prediction: int
    confidence: float
    risk_score: float

    def risk_label(self) -> str:
        if self.risk_score < 0.3:
            return "BENIGN"
        if self.risk_score < 0.6:
            return "SUSPICIOUS"
        return "MALICIOUS"


def explain_prediction(
    model: MalwareGAT,
    data: Data,
    config: ExplainConfig | None = None,
) -> ExplanationResult:
    config = config or ExplainConfig()
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)
        prediction = out.argmax(dim=1).item()
        confidence = probs[0, prediction].item()
        risk_score = probs[0, 1].item()

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=config.explainer_epochs, lr=config.explainer_lr),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
    )

    node_names = data.node_names if hasattr(data, "node_names") else [
        f"node_{i}" for i in range(data.num_nodes)
    ]

    node_mask = explanation.node_mask
    if node_mask is not None:
        node_scores = node_mask.mean(dim=1).detach().cpu()
    else:
        node_scores = torch.zeros(data.num_nodes)

    node_importances = {}
    for i, name in enumerate(node_names):
        node_importances[name] = float(node_scores[i])

    sorted_nodes = sorted(node_importances.items(), key=lambda x: x[1], reverse=True)
    top_nodes = sorted_nodes[:config.top_k_nodes]

    edge_importances = []
    edge_mask = explanation.edge_mask
    if edge_mask is not None:
        edge_scores = edge_mask.detach().cpu()
        for i in range(data.edge_index.size(1)):
            src_idx = data.edge_index[0, i].item()
            dst_idx = data.edge_index[1, i].item()
            src_name = node_names[src_idx] if src_idx < len(node_names) else f"node_{src_idx}"
            dst_name = node_names[dst_idx] if dst_idx < len(node_names) else f"node_{dst_idx}"
            edge_importances.append((src_name, dst_name, float(edge_scores[i])))

    edge_importances.sort(key=lambda x: x[2], reverse=True)

    return ExplanationResult(
        node_importances=node_importances,
        top_nodes=top_nodes,
        edge_importances=edge_importances[:30],
        prediction=prediction,
        confidence=confidence,
        risk_score=risk_score,
    )
