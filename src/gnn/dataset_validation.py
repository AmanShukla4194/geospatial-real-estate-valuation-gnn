"""
PyTorch Geometric Dataset Validation.

Validates the PyTorch Geometric Data object created for Graph
Neural Network training.

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from __future__ import annotations

import torch

from src.config import (
    NODE_FEATURE_COLUMNS,
    GRAPH_TARGET_COLUMN,
)

from src.gnn.graph_dataset import create_graph_data


# =============================================================================
# Validation
# =============================================================================

def validate_dataset():

    print("=" * 70)
    print("Validating PyTorch Geometric Dataset")
    print("=" * 70)

    graph = create_graph_data()

    print()

    print("Running validation checks...\n")

    # ---------------------------------------------------------

    if graph.x is None:
        raise ValueError("Node feature matrix is missing.")

    print("✓ Node feature matrix exists")

    # ---------------------------------------------------------

    if graph.edge_index is None:
        raise ValueError("Edge index is missing.")

    print("✓ Edge index exists")

    # ---------------------------------------------------------

    if graph.y is None:
        raise ValueError("Target tensor is missing.")

    print("✓ Target tensor exists")

    # ---------------------------------------------------------

    expected_features = len(
        NODE_FEATURE_COLUMNS
    )

    if graph.num_node_features != expected_features:

        raise ValueError(
            f"Expected {expected_features} node features "
            f"but found {graph.num_node_features}."
        )

    print("✓ Feature dimension validation passed")

    # ---------------------------------------------------------

    if graph.num_nodes != graph.y.shape[0]:

        raise ValueError(
            "Number of target values does not match "
            "number of nodes."
        )

    print("✓ Target dimension validation passed")

    # ---------------------------------------------------------

    if graph.edge_index.shape[0] != 2:

        raise ValueError(
            "Edge index should have shape [2, num_edges]."
        )

    print("✓ Edge index shape validation passed")

    # ---------------------------------------------------------

    if torch.isnan(graph.x).any():

        raise ValueError(
            "NaN detected in node features."
        )

    print("✓ Node feature NaN validation passed")

    # ---------------------------------------------------------

    if torch.isnan(graph.y).any():

        raise ValueError(
            "NaN detected in target tensor."
        )

    print("✓ Target NaN validation passed")

    # ---------------------------------------------------------

    if torch.isnan(
        graph.edge_index.float()
    ).any():

        raise ValueError(
            "NaN detected in edge index."
        )

    print("✓ Edge index NaN validation passed")

    print()

    print("Dataset Statistics")
    print("-" * 70)

    print(f"Nodes              : {graph.num_nodes:,}")

    print(f"Edges              : {graph.num_edges:,}")

    print(f"Node Features      : {graph.num_node_features}")

    print(f"Target Column      : {GRAPH_TARGET_COLUMN}")

    print(f"Tensor Device      : {graph.x.device}")

    print(f"Tensor Type        : {graph.x.dtype}")

    print("-" * 70)

    print()

    print("Graph dataset validation completed successfully.")

    print("=" * 70)

    return graph


if __name__ == "__main__":

    validate_dataset()