"""
Spatial Graph Builder.

Constructs a K-Nearest Neighbor (KNN) graph from the engineered
housing dataset.

Each property is represented as one graph node and connected to its
K nearest neighbouring properties based on geographical coordinates.

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

from src.config import (
    PROCESSED_DATA_DIR,
    K_NEAREST_NEIGHBORS,
)

# =============================================================================
# Paths
# =============================================================================

INPUT_DATASET = (
    PROCESSED_DATA_DIR
    / "engineered_housing_data.csv"
)

GRAPH_DIR = Path("graph")

GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EDGE_INDEX_FILE = GRAPH_DIR / "edge_index.npy"

EDGE_DISTANCE_FILE = GRAPH_DIR / "edge_distance.npy"

NODE_MAPPING_FILE = GRAPH_DIR / "node_mapping.joblib"


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset() -> pd.DataFrame:

    if not INPUT_DATASET.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_DATASET}"
        )

    dataframe = pd.read_csv(
        INPUT_DATASET
    )

    return dataframe


# =============================================================================
# Node Mapping
# =============================================================================

def create_node_mapping(
    dataframe: pd.DataFrame,
):

    mapping = {
        index: index
        for index in dataframe.index
    }

    return mapping


# =============================================================================
# Graph Construction
# =============================================================================

def build_knn_graph(
    dataframe: pd.DataFrame,
):

    coordinates = dataframe[
        [
            "lat",
            "long",
        ]
    ].to_numpy()

    knn = NearestNeighbors(

        n_neighbors=K_NEAREST_NEIGHBORS + 1,

        metric="euclidean",
    )

    knn.fit(
        coordinates
    )

    distances, indices = knn.kneighbors(
        coordinates
    )

    edge_index = []

    edge_distance = []

    for node in range(
        len(indices)
    ):

        for neighbour_position in range(
            1,
            K_NEAREST_NEIGHBORS + 1,
        ):

            neighbour = indices[
                node,
                neighbour_position,
            ]

            edge_index.append(
                [
                    node,
                    neighbour,
                ]
            )

            edge_distance.append(
                distances[
                    node,
                    neighbour_position,
                ]
            )

    edge_index = np.array(
        edge_index,
        dtype=np.int64,
    )

    edge_distance = np.array(
        edge_distance,
        dtype=np.float32,
    )

    return (
        edge_index,
        edge_distance,
    )


# =============================================================================
# Save Graph
# =============================================================================

def save_graph(
    edge_index,
    edge_distance,
    node_mapping,
):

    np.save(
        EDGE_INDEX_FILE,
        edge_index,
    )

    np.save(
        EDGE_DISTANCE_FILE,
        edge_distance,
    )

    joblib.dump(
        node_mapping,
        NODE_MAPPING_FILE,
    )


# =============================================================================
# Main Pipeline
# =============================================================================

def build_graph():

    print("=" * 70)
    print("Building Spatial KNN Graph")
    print("=" * 70)

    dataframe = load_dataset()

    print(
        f"Dataset loaded: {len(dataframe):,} houses"
    )

    node_mapping = create_node_mapping(
        dataframe
    )

    (
        edge_index,
        edge_distance,
    ) = build_knn_graph(
        dataframe
    )

    save_graph(
        edge_index,
        edge_distance,
        node_mapping,
    )

    print()

    print(
        f"Nodes : {len(node_mapping):,}"
    )

    print(
        f"Edges : {len(edge_index):,}"
    )

    print(
        f"K     : {K_NEAREST_NEIGHBORS}"
    )

    print()

    print(
        "Graph artifacts saved."
    )

    print()

    print(
        EDGE_INDEX_FILE
    )

    print(
        EDGE_DISTANCE_FILE
    )

    print(
        NODE_MAPPING_FILE
    )

    print("=" * 70)


if __name__ == "__main__":

    build_graph()