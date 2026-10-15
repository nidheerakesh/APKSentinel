from __future__ import annotations

from dataclasses import dataclass
from torch_geometric.data import Data

from .explainer import ExplanationResult


@dataclass
class AgentAttribution:
    agent_name: str
    contribution_pct: float
    top_entities: list[tuple[str, float]]
    evidence_count: int


def attribute_to_agents(
    explanation: ExplanationResult,
    data: Data,
) -> list[AgentAttribution]:
    source_agents = getattr(data, "source_agents", [])
    node_names = getattr(data, "node_names", [])

    if not source_agents or not node_names:
        return []

    agent_scores: dict[str, float] = {}
    agent_entities: dict[str, list[tuple[str, float]]] = {}
    agent_counts: dict[str, int] = {}

    for name, score in explanation.node_importances.items():
        try:
            idx = node_names.index(name)
        except ValueError:
            continue

        if idx >= len(source_agents):
            continue

        agent = source_agents[idx]
        if not agent or agent == "none":
            continue

        agent_scores[agent] = agent_scores.get(agent, 0.0) + score
        agent_entities.setdefault(agent, []).append((name, score))
        agent_counts[agent] = agent_counts.get(agent, 0) + 1

    total_score = sum(agent_scores.values())
    if total_score == 0:
        total_score = 1.0

    attributions = []
    for agent_name in sorted(agent_scores, key=agent_scores.get, reverse=True):
        entities = sorted(agent_entities[agent_name], key=lambda x: x[1], reverse=True)
        attributions.append(AgentAttribution(
            agent_name=agent_name,
            contribution_pct=round(100 * agent_scores[agent_name] / total_score, 1),
            top_entities=entities[:5],
            evidence_count=agent_counts[agent_name],
        ))

    return attributions


def generate_explanation_text(
    explanation: ExplanationResult,
    attributions: list[AgentAttribution],
) -> str:
    label = explanation.risk_label()
    score = explanation.risk_score

    parts = [f"Classification: **{label}** (risk score: {score:.2f}, confidence: {explanation.confidence:.2f})"]

    if explanation.top_nodes:
        indicators = []
        for name, imp in explanation.top_nodes[:5]:
            short_name = name.split(";->")[-1] if ";->" in name else name.split(":")[-1]
            if len(short_name) > 60:
                short_name = short_name[:57] + "..."
            indicators.append(f"  - {short_name} (importance: {imp:.3f})")
        parts.append("Key indicators:\n" + "\n".join(indicators))

    if attributions:
        attr_lines = []
        for a in attributions[:4]:
            attr_lines.append(f"  - **{a.agent_name}**: {a.contribution_pct}% ({a.evidence_count} items)")
        parts.append("Agent contributions:\n" + "\n".join(attr_lines))

    return "\n\n".join(parts)
