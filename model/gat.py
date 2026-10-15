from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


NUM_NODE_TYPES = 6


class MalwareGAT(nn.Module):
    def __init__(
        self,
        in_channels: int = 25,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout
        self.in_channels = in_channels
        aug_channels = in_channels + 2

        self.conv1 = GATConv(
            aug_channels, hidden_channels, heads=num_heads, dropout=dropout
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)

        self.conv2 = GATConv(
            hidden_channels * num_heads, hidden_channels,
            heads=1, concat=False, dropout=dropout
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        graph_feat_dim = 5 + NUM_NODE_TYPES + 3 + 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 + graph_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def _augment_features(self, x, edge_index):
        n = x.size(0)
        in_deg = torch.zeros(n, device=x.device)
        out_deg = torch.zeros(n, device=x.device)
        if edge_index.size(1) > 0:
            ones = torch.ones(edge_index.size(1), device=x.device)
            in_deg.scatter_add_(0, edge_index[1], ones)
            out_deg.scatter_add_(0, edge_index[0], ones)
        return torch.cat([x, torch.log1p(in_deg).unsqueeze(1), torch.log1p(out_deg).unsqueeze(1)], dim=1)

    def _graph_features(self, x, edge_index, batch):
        num_graphs = batch.max().item() + 1
        dev = x.device
        num_nodes = torch.zeros(num_graphs, device=dev)
        num_nodes.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))

        num_edges = torch.zeros(num_graphs, device=dev)
        if edge_index.size(1) > 0:
            edge_batch = batch[edge_index[0]]
            num_edges.scatter_add_(0, edge_batch, torch.ones(edge_index.size(1), device=dev))
        density = num_edges / (num_nodes * (num_nodes - 1) + 1e-6)

        avg_feat = torch.zeros(num_graphs, device=dev)
        feat_mean = x.mean(dim=1)
        avg_feat.scatter_add_(0, batch, feat_mean)
        avg_feat = avg_feat / (num_nodes + 1e-6)

        feat_std = torch.zeros(num_graphs, device=dev)
        feat_std_vals = x.std(dim=1)
        feat_std.scatter_add_(0, batch, feat_std_vals)
        feat_std = feat_std / (num_nodes + 1e-6)

        type_counts = torch.zeros(num_graphs, NUM_NODE_TYPES, device=dev)
        node_types = x[:, :NUM_NODE_TYPES].argmax(dim=1)
        for t in range(NUM_NODE_TYPES):
            mask = (node_types == t).float()
            type_counts[:, t].scatter_add_(0, batch, mask)
        type_ratios = type_counts / (num_nodes.unsqueeze(1) + 1e-6)

        n = x.size(0)
        in_deg = torch.zeros(n, device=dev)
        out_deg = torch.zeros(n, device=dev)
        if edge_index.size(1) > 0:
            ones = torch.ones(edge_index.size(1), device=dev)
            in_deg.scatter_add_(0, edge_index[1], ones)
            out_deg.scatter_add_(0, edge_index[0], ones)
        total_deg = in_deg + out_deg

        mean_deg = torch.zeros(num_graphs, device=dev)
        mean_deg.scatter_add_(0, batch, total_deg)
        mean_deg = mean_deg / (num_nodes + 1e-6)

        max_deg = torch.zeros(num_graphs, device=dev)
        for i in range(num_graphs):
            mask = (batch == i)
            if mask.any():
                max_deg[i] = total_deg[mask].max()

        std_deg = torch.zeros(num_graphs, device=dev)
        for i in range(num_graphs):
            mask = (batch == i)
            if mask.any() and mask.sum() > 1:
                std_deg[i] = total_deg[mask].std()

        nonzero_feats = (x[:, NUM_NODE_TYPES:] != 0).float().mean(dim=1)
        activation_rate = torch.zeros(num_graphs, device=dev)
        activation_rate.scatter_add_(0, batch, nonzero_feats)
        activation_rate = activation_rate / (num_nodes + 1e-6)

        return torch.cat([
            torch.log1p(num_nodes).unsqueeze(1),
            torch.log1p(num_edges).unsqueeze(1),
            density.unsqueeze(1),
            avg_feat.unsqueeze(1),
            feat_std.unsqueeze(1),
            type_ratios,
            torch.log1p(mean_deg).unsqueeze(1),
            torch.log1p(max_deg).unsqueeze(1),
            std_deg.unsqueeze(1),
            activation_rate.unsqueeze(1),
        ], dim=1)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        graph_feats = self._graph_features(x, edge_index, batch)
        x = self._augment_features(x, edge_index)

        x = self.bn1(F.elu(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn2(self.conv2(x, edge_index))

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max, graph_feats], dim=1)

        return self.classifier(x)

    def get_node_embeddings(self, x, edge_index):
        x = self._augment_features(x, edge_index)
        x = self.bn1(F.elu(self.conv1(x, edge_index)))
        x = self.bn2(self.conv2(x, edge_index))
        return x
