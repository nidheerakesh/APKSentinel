from __future__ import annotations

import networkx as nx
import torch
from torch_geometric.data import Data

from agents.base import EvidenceItem
from .schema import NodeType, EdgeType, EVIDENCE_TYPE_TO_NODE_TYPE, infer_edge_type
from .features import encode_node_features


class GraphBuilder:
    def build(self, evidence: list[EvidenceItem], label: int = -1) -> Data:
        G = nx.DiGraph()

        for item in evidence:
            node_type = EVIDENCE_TYPE_TO_NODE_TYPE.get(
                item.evidence_type.value, NodeType.METHOD
            )
            G.add_node(item.entity, node_type=node_type,
                       source_agent=item.source_agent,
                       attributes=item.attributes,
                       confidence=item.confidence)

            for related in item.related_entities:
                if related not in G:
                    rel_type = self._infer_related_type(related, node_type)
                    G.add_node(related, node_type=rel_type,
                               source_agent=item.source_agent,
                               attributes={}, confidence=1.0)

                src_type = G.nodes[item.entity]["node_type"]
                dst_type = G.nodes[related]["node_type"]
                edge_type = infer_edge_type(src_type, dst_type)
                G.add_edge(item.entity, related, edge_type=edge_type)

        if len(G) == 0:
            return self._empty_graph(label)

        node_list = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(node_list)}

        x = torch.stack([
            encode_node_features(
                G.nodes[n]["node_type"],
                G.nodes[n].get("attributes", {}),
            )
            for n in node_list
        ])

        edges = list(G.edges())
        if edges:
            src_indices = [node_to_idx[e[0]] for e in edges]
            dst_indices = [node_to_idx[e[1]] for e in edges]
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        node_types = torch.tensor(
            [int(G.nodes[n]["node_type"]) for n in node_list], dtype=torch.long
        )
        source_agents = [G.nodes[n].get("source_agent", "") for n in node_list]

        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long) if label >= 0 else None,
            num_nodes=len(node_list),
        )
        data.node_types = node_types
        data.node_names = node_list
        data.source_agents = source_agents

        return data

    @staticmethod
    def _infer_related_type(entity: str, parent_type: NodeType) -> NodeType:
        entity_lower = entity.lower()
        if "permission" in entity_lower or entity.startswith("android.permission"):
            return NodeType.PERMISSION
        if "http" in entity_lower or "://" in entity_lower:
            return NodeType.NETWORK_ENDPOINT
        if "crypto" in entity_lower or "cipher" in entity_lower or "javax/crypto" in entity_lower:
            return NodeType.CRYPTO_OP
        if ";->" in entity:
            return NodeType.METHOD
        if entity.startswith("L") and entity.endswith(";"):
            return NodeType.CLASS
        return NodeType.METHOD

    @staticmethod
    def _empty_graph(label: int) -> Data:
        return Data(
            x=torch.zeros((1, 25)),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            y=torch.tensor([label], dtype=torch.long) if label >= 0 else None,
            num_nodes=1,
            node_types=torch.tensor([0], dtype=torch.long),
            node_names=["empty"],
            source_agents=["none"],
        )
