"""
Graph Neural Network Evaluation.

Evaluates the trained GNN model on the test set.

Author:
    Aman Shukla
"""

from __future__ import annotations

import numpy as np
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

from src.config import (
    GNN_MODEL_PATH,
)

from src.gnn.graph_dataset import (
    build_graph_dataset,
)

from src.models.gnn_model import (
    PropertyValueGNN,
)


def evaluate_gnn():

    print("=" * 70)
    print("Evaluating Graph Neural Network")
    print("=" * 70)

    graph = build_graph_dataset()

    model = PropertyValueGNN(
        input_channels=graph.num_node_features,
    )

    model.load_state_dict(
        torch.load(
            GNN_MODEL_PATH,
            map_location="cpu",
        )
    )

    model.eval()

    with torch.no_grad():

        prediction = model(graph).squeeze()

    prediction = prediction[
        graph.test_mask
    ].cpu().numpy()

    target = graph.y[
        graph.test_mask
    ].cpu().numpy()

    mae = mean_absolute_error(
        target,
        prediction,
    )

    rmse = np.sqrt(
        mean_squared_error(
            target,
            prediction,
        )
    )

    mape = (
        mean_absolute_percentage_error(
            target,
            prediction,
        )
        * 100
    )

    r2 = r2_score(
        target,
        prediction,
    )

    print()

    print(f"MAE  : {mae:,.2f}")

    print(f"RMSE : {rmse:,.2f}")

    print(f"MAPE : {mape:.2f}%")

    print(f"R²   : {r2:.4f}")

    print()

    print("=" * 70)

    return {

        "MAE": mae,

        "RMSE": rmse,

        "MAPE": mape,

        "R2": r2,
    }


if __name__ == "__main__":

    evaluate_gnn()