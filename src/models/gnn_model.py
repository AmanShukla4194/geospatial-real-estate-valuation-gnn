"""
Graph Neural Network Architecture.

Defines the Graph Convolutional Network (GCN) used for
property valuation.

Author:
    Aman Shukla
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torch.nn import Linear

from torch_geometric.nn import (
    GCNConv,
)

from src.config import (
    GNN_HIDDEN_CHANNELS,
    GNN_OUTPUT_CHANNELS,
)


class PropertyValueGNN(torch.nn.Module):
    """
    Graph Neural Network for property price prediction.
    """

    def __init__(
        self,
        input_channels: int,
    ):

        super().__init__()

        self.conv1 = GCNConv(
            input_channels,
            GNN_HIDDEN_CHANNELS,
        )

        self.conv2 = GCNConv(
            GNN_HIDDEN_CHANNELS,
            GNN_HIDDEN_CHANNELS,
        )

        self.output_layer = Linear(
            GNN_HIDDEN_CHANNELS,
            GNN_OUTPUT_CHANNELS,
        )

    def forward(
        self,
        data,
    ):

        x = data.x
        edge_index = data.edge_index

        x = self.conv1(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=0.30,
            training=self.training,
        )

        x = self.conv2(
            x,
            edge_index,
        )

        x = F.relu(x)

        predictions = self.output_layer(
            x,
        )

        return predictions