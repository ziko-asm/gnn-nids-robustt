"""
GNN architectures for node-level IoT intrusion detection.

Supported `model_type` values:
    mlp   - feature-only baseline (ignores the graph). Essential control:
            if a GNN does not beat this, the graph is not helping.
    gcn   - Kipf & Welling, 2017. Supports optional edge weights.
    gat   - Velickovic et al., 2018 (multi-head attention).
    sage  - Hamilton et al., 2017 (GraphSAGE, mean aggregator).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

MODEL_TYPES = ("mlp", "gcn", "gat", "sage")


class IoTGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        model_type: str = "gcn",
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = model_type.lower()
        self.dropout = dropout

        if self.model_type == "mlp":
            self.conv1 = nn.Linear(in_channels, hidden_channels)
            self.conv2 = nn.Linear(hidden_channels, out_channels)
        elif self.model_type == "gcn":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif self.model_type == "gat":
            per_head = max(1, hidden_channels // heads)
            self.conv1 = GATConv(in_channels, per_head, heads=heads, concat=True, dropout=dropout)
            self.conv2 = GATConv(per_head * heads, out_channels, heads=1, concat=False, dropout=dropout)
        elif self.model_type == "sage":
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        else:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. Choose one of: {', '.join(MODEL_TYPES)}"
            )

    def _layer(self, conv, x, edge_index, edge_weight):
        if self.model_type == "mlp":
            return conv(x)
        if self.model_type == "gcn":
            return conv(x, edge_index, edge_weight)
        return conv(x, edge_index)  # GAT / SAGE ignore scalar edge weights

    def forward(self, x, edge_index, edge_weight=None):
        """Returns log-probabilities of shape [num_nodes, out_channels]."""
        x = F.dropout(x, p=self.dropout * 0.4, training=self.training)
        x = self._layer(self.conv1, x, edge_index, edge_weight)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self._layer(self.conv2, x, edge_index, edge_weight)
        return F.log_softmax(x, dim=-1)
