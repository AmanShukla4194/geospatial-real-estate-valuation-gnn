"""
Map Service.

Provides reusable map utilities for the Streamlit dashboard.

Author:
    Aman Shukla
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
)

# ==========================================================
# Dataset Path
# ==========================================================

ENGINEERED_DATASET = (
    PROCESSED_DATA_DIR
    / "engineered_housing_data.csv"
)


# ==========================================================
# Load Map Dataset
# ==========================================================

def load_map_dataset():

    dataframe = pd.read_csv(
        ENGINEERED_DATASET
    )

    return dataframe