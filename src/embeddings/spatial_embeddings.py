"""
Spatial Embedding Generator.

Creates the initial spatial node feature matrix used as the
input representation for the Graph Neural Network.

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    PROCESSED_DATA_DIR,
    NODE_FEATURE_COLUMNS,
)

ENGINEERED_DATA_FILE = (
    PROCESSED_DATA_DIR
    / "engineered_housing_data.csv"
)


def load_dataset() -> pd.DataFrame:

    dataframe = pd.read_csv(
        ENGINEERED_DATA_FILE
    )

    return dataframe


def build_spatial_embeddings():

    print("=" * 70)
    print("Generating Spatial Feature Embeddings")
    print("=" * 70)

    dataframe = load_dataset()

    feature_columns = list(dict.fromkeys(
        NODE_FEATURE_COLUMNS
        + [
            "lat",
            "long",
        ]
    ))

    feature_matrix = dataframe[
        feature_columns
    ].copy()

    scaler = StandardScaler()

    embeddings = scaler.fit_transform(
        feature_matrix
    )

    print()

    print(
        f"Nodes              : {embeddings.shape[0]:,}"
    )

    print(
        f"Embedding Dimension: {embeddings.shape[1]}"
    )

    print()

    print(
        "Spatial feature embeddings generated."
    )

    print("=" * 70)

    return embeddings


if __name__ == "__main__":

    build_spatial_embeddings()