"""
Graph Neural Network Training Pipeline.

Author:
    Aman Shukla
"""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F

from torch.optim import Adam

from src.config import (
    GNN_EPOCHS,
    GNN_LEARNING_RATE,
    GNN_WEIGHT_DECAY,
    GNN_MODEL_PATH,
    TRAINING_HISTORY_PATH,
)

from src.gnn.graph_dataset import (
    build_graph_dataset,
)

from src.models.gnn_model import (
    PropertyValueGNN,
)


def train_gnn():

    print("=" * 70)
    print("Training Graph Neural Network")
    print("=" * 70)

    graph = build_graph_dataset()

    model = PropertyValueGNN(
        input_channels=graph.num_node_features
    )

    optimizer = Adam(
        model.parameters(),
        lr=GNN_LEARNING_RATE,
        weight_decay=GNN_WEIGHT_DECAY,
    )

    loss_history = []

    model.train()

    for epoch in range(
        1,
        GNN_EPOCHS + 1,
    ):

        optimizer.zero_grad()

        prediction = model(
            graph
        ).squeeze()

        loss = F.mse_loss(
            prediction[
                graph.train_mask
            ],
            graph.y[
                graph.train_mask
            ],
        )

        loss.backward()

        optimizer.step()

        loss_history.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
            }
        )

        if epoch % 10 == 0:

            print(
                f"Epoch {epoch:03d} | Loss = {loss.item():.4f}"
            )

    torch.save(
        model.state_dict(),
        GNN_MODEL_PATH,
    )

    pd.DataFrame(
        loss_history
    ).to_csv(
        TRAINING_HISTORY_PATH,
        index=False,
    )

    print()

    print("Training completed successfully.")

    print()

    print("Model saved to:")
    print(GNN_MODEL_PATH)

    print()

    print("Training history saved to:")
    print(TRAINING_HISTORY_PATH)

    print("=" * 70)


if __name__ == "__main__":

    train_gnn()