"""
Graph Validation Module.

This module validates the spatial K-Nearest Neighbor graph generated
for the Geospatial Real Estate Valuation project.

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.config import K_NEAREST_NEIGHBORS

# =============================================================================
# Paths
# =============================================================================

GRAPH_DIR = Path("graph")

EDGE_INDEX_FILE = GRAPH_DIR / "edge_index.npy"
EDGE_DISTANCE_FILE = GRAPH_DIR / "edge_distance.npy"
NODE_MAPPING_FILE = GRAPH_DIR / "node_mapping.joblib"


# =============================================================================
# Load Graph Artifacts
# =============================================================================

def load_graph():

    if not EDGE_INDEX_FILE.exists():
        raise FileNotFoundError(EDGE_INDEX_FILE)

    if not EDGE_DISTANCE_FILE.exists():
        raise FileNotFoundError(EDGE_DISTANCE_FILE)

    if not NODE_MAPPING_FILE.exists():
        raise FileNotFoundError(NODE_MAPPING_FILE)

    edge_index = np.load(
        EDGE_INDEX_FILE
    )

    edge_distance = np.load(
        EDGE_DISTANCE_FILE
    )

    node_mapping = joblib.load(
        NODE_MAPPING_FILE
    )

    return (
        edge_index,
        edge_distance,
        node_mapping,
    )


# =============================================================================
# Validation
# =============================================================================

def validate_graph():

    print("=" * 70)
    print("Validating Spatial Graph")
    print("=" * 70)

    (
        edge_index,
        edge_distance,
        node_mapping,
    ) = load_graph()

    number_of_nodes = len(
        node_mapping
    )

    number_of_edges = len(
        edge_index
    )

    expected_edges = (
        number_of_nodes
        * K_NEAREST_NEIGHBORS
    )

    print(
        f"Nodes             : {number_of_nodes:,}"
    )

    print(
        f"Edges             : {number_of_edges:,}"
    )

    print(
        f"Expected Edges    : {expected_edges:,}"
    )

    if number_of_edges != expected_edges:

        raise ValueError(
            "Edge count validation failed."
        )

    print(
        "\nEdge count validation passed."
    )

    if len(edge_distance) != number_of_edges:

        raise ValueError(
            "Edge distance length mismatch."
        )

    print(
        "Distance validation passed."
    )

    if edge_index.shape[1] != 2:

        raise ValueError(
            "Edge index should contain source and destination."
        )

    print(
        "Edge index format validation passed."
    )

    if np.any(edge_distance < 0):

        raise ValueError(
            "Negative distances detected."
        )

    print(
        "Distance range validation passed."
    )

    degrees = np.bincount(
        edge_index[:, 0],
        minlength=number_of_nodes,
    )

    print()

    print("Degree Statistics")
    print("-" * 70)

    print(
        f"Minimum Degree : {degrees.min()}"
    )

    print(
        f"Maximum Degree : {degrees.max()}"
    )

    print(
        f"Average Degree : {degrees.mean():.2f}"
    )

    print("-" * 70)

    if degrees.min() != K_NEAREST_NEIGHBORS:

        raise ValueError(
            "Unexpected node degree."
        )

    print()

    print(
        "Graph validation completed successfully."
    )

    print("=" * 70)

    return {
        "nodes": number_of_nodes,
        "edges": number_of_edges,
        "degree_mean": degrees.mean(),
        "degree_min": degrees.min(),
        "degree_max": degrees.max(),
    }


if __name__ == "__main__":

    validate_graph()