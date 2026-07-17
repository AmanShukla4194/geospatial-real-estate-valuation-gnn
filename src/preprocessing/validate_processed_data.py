"""
Processed Dataset Validation.

This module validates the cleaned dataset before feature engineering.

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR

# =============================================================================
# Configuration
# =============================================================================

DATASET_PATH = PROCESSED_DATA_DIR / "clean_housing_data.csv"

REQUIRED_COLUMNS = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "lat",
    "long",
]

# =============================================================================
# Validation
# =============================================================================


def validate_dataset():

    print("=" * 70)
    print("Processed Dataset Validation")
    print("=" * 70)

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Processed dataset not found:\n{DATASET_PATH}"
        )

    dataframe = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully.\n")

    print(f"Rows    : {len(dataframe)}")
    print(f"Columns : {len(dataframe.columns)}")

    print("\nChecking required columns...")

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    print("All required columns exist.")

    print("\nChecking missing values...")

    missing_values = dataframe.isnull().sum().sum()

    print(f"Total Missing Values : {missing_values}")

    print("\nChecking duplicate rows...")

    duplicates = dataframe.duplicated().sum()

    print(f"Duplicate Rows : {duplicates}")

    print("\nChecking coordinate validity...")

    invalid_coordinates = dataframe[
        (dataframe["lat"] < -90)
        | (dataframe["lat"] > 90)
        | (dataframe["long"] < -180)
        | (dataframe["long"] > 180)
    ]

    print(f"Invalid Coordinates : {len(invalid_coordinates)}")

    print("\nValidation completed successfully.")

    print("=" * 70)


if __name__ == "__main__":
    validate_dataset()