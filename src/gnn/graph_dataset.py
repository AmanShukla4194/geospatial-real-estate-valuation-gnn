"""
PyTorch Geometric Graph Dataset Builder.

This module converts the engineered housing dataset and the
constructed K-Nearest Neighbor graph into a PyTorch Geometric
Data object.

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
import torch

from torch_geometric.data import Data

from src.config import (
    PROCESSED_DATA_DIR,
    NODE_FEATURE_COLUMNS,
    GRAPH_TARGET_COLUMN,
)

# =============================================================================
# Paths
# =============================================================================

GRAPH_DIR = Path("graph")

ENGINEERED_DATASET = (
    PROCESSED_DATA_DIR
    / "engineered_housing_data.csv"
)

EDGE_INDEX_FILE = GRAPH_DIR / "edge_index.npy"

NODE_MAPPING_FILE = GRAPH_DIR / "node_mapping.joblib"


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset():

    dataframe = pd.read_csv(
        ENGINEERED_DATASET
    )

    return dataframe


def load_graph():

    edge_index = np.load(
        EDGE_INDEX_FILE
    )

    node_mapping = joblib.load(
        NODE_MAPPING_FILE
    )

    return (
        edge_index,
        node_mapping,
    )


# =============================================================================
# Node Features
# =============================================================================

def build_node_features(
    dataframe,
):

    node_features = dataframe[
        NODE_FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    return torch.tensor(
        node_features,
        dtype=torch.float,
    )


# =============================================================================
# Target Vector
# =============================================================================

def build_targets(
    dataframe,
):

    target = dataframe[
        GRAPH_TARGET_COLUMN
    ].to_numpy(
        dtype=np.float32
    )

    return torch.tensor(
        target,
        dtype=torch.float,
    )


# =============================================================================
# Edge Index
# =============================================================================

def build_edge_index(
    edge_index,
):

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long,
    )

    edge_index = edge_index.t().contiguous()

    return edge_index


# =============================================================================
# Graph Object
# =============================================================================

def create_graph_data():

    print("=" * 70)

    print(
        "Preparing PyTorch Geometric Dataset"
    )

    print("=" * 70)

    dataframe = load_dataset()

    edge_index, node_mapping = load_graph()

    x = build_node_features(
        dataframe
    )

    y = build_targets(
        dataframe
    )

    edge_index = build_edge_index(
        edge_index
    )

    graph = Data(

        x=x,

        edge_index=edge_index,

        y=y,
    )

    print()

    print(
        f"Nodes        : {graph.num_nodes:,}"
    )

    print(
        f"Edges        : {graph.num_edges:,}"
    )

    print(
        f"Features     : {graph.num_node_features}"
    )

    print(
        f"Target Shape : {graph.y.shape}"
    )

    print()

    print(
        "PyTorch Geometric Data object created."
    )

    print("=" * 70)

    return graph


if __name__ == "__main__":

    create_graph_data()