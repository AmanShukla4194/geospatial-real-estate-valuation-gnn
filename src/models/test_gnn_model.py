"""
Quick architecture validation.

Author:
    Aman Shukla
"""

import torch

from torch_geometric.data import Data

from src.models.gnn_model import (
    PropertyValueGNN,
)

x = torch.randn(
    100,
    8,
)

edge_index = torch.randint(
    0,
    100,
    (
        2,
        500,
    ),
)

data = Data(
    x=x,
    edge_index=edge_index,
)

model = PropertyValueGNN(
    input_channels=8,
)

prediction = model(
    data,
)

print("=" * 70)

print("Graph Neural Network Test")

print("=" * 70)

print(
    "Prediction Shape:",
    prediction.shape,
)

print("=" * 70)