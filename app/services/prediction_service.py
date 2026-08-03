"""
Prediction Service.

Provides prediction-related utilities for the Streamlit dashboard.

Author:
    Aman Shukla
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
)

# ==========================================================
# Dataset
# ==========================================================

ENGINEERED_DATASET = (
    PROCESSED_DATA_DIR
    / "engineered_housing_data.csv"
)


# ==========================================================
# Load Prediction Dataset
# ==========================================================

def load_prediction_dataset():

    dataframe = pd.read_csv(
        ENGINEERED_DATASET
    )

    return dataframe


# ==========================================================
# Prediction Summary
# ==========================================================

def get_prediction_summary():

    dataframe = load_prediction_dataset()

    return {

        "Average Price":
            dataframe["price"].mean(),

        "Median Price":
            dataframe["price"].median(),

        "Minimum Price":
            dataframe["price"].min(),

        "Maximum Price":
            dataframe["price"].max(),

    }