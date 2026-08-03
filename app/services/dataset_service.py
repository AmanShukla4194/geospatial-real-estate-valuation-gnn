"""
Dataset Service.

Provides reusable dataset functions for the Streamlit dashboard.

Author:
    Aman Shukla
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
)

# =============================================================================
# Dataset Path
# =============================================================================

ENGINEERED_DATASET = (
    PROCESSED_DATA_DIR
    / "engineered_housing_data.csv"
)

# =============================================================================
# Load Dataset
# =============================================================================


def load_dataset():

    dataframe = pd.read_csv(
        ENGINEERED_DATASET
    )

    return dataframe


# =============================================================================
# Dataset Statistics
# =============================================================================


def get_dataset_statistics():

    dataframe = load_dataset()

    statistics = {

        "Rows": len(dataframe),

        "Columns": len(dataframe.columns),

        "Features": len(dataframe.columns) - 1,

        "Target": "price",

    }

    return statistics


# =============================================================================
# Feature List
# =============================================================================


def get_feature_names():

    dataframe = load_dataset()

    return list(
        dataframe.columns
    )