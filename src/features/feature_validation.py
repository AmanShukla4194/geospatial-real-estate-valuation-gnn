"""
Feature Validation Module.

This module validates the engineered features generated for the
baseline machine-learning stage of the Geospatial Real Estate
Valuation project.

The validation process checks:

- Dataset availability
- Required engineered features
- Missing values
- Infinite values
- Numeric data types
- Logical feature ranges
- Coordinate-derived distance values
- Summary statistics

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR


ENGINEERED_DATA_FILE = (
    PROCESSED_DATA_DIR
    / "engineered_housing_data.csv"
)


ENGINEERED_FEATURES = [
    "sale_year",
    "sale_month",
    "house_age",
    "is_renovated",
    "years_since_renovation",
    "distance_to_city_center_km",
]


def load_engineered_dataset() -> pd.DataFrame:
    """
    Load the feature-engineered housing dataset.

    Returns
    -------
    pd.DataFrame
        Feature-engineered housing dataset.

    Raises
    ------
    FileNotFoundError
        If the engineered dataset does not exist.
    """

    if not ENGINEERED_DATA_FILE.exists():

        raise FileNotFoundError(
            "Feature-engineered dataset was not found.\n"
            f"Expected location: {ENGINEERED_DATA_FILE}\n"
            "Run the feature engineering pipeline first."
        )

    dataframe = pd.read_csv(
        ENGINEERED_DATA_FILE
    )

    print(
        "Engineered dataset loaded: "
        f"{len(dataframe):,} rows, "
        f"{len(dataframe.columns)} columns."
    )

    return dataframe


def validate_required_features(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate that all required engineered features exist.
    """

    missing_features = [
        feature
        for feature in ENGINEERED_FEATURES
        if feature not in dataframe.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing engineered feature(s): "
            + ", ".join(missing_features)
        )

    print(
        "Required engineered feature validation passed."
    )


def validate_missing_values(
    dataframe: pd.DataFrame,
) -> None:
    """
    Ensure engineered features contain no missing values.
    """

    missing_counts = (
        dataframe[ENGINEERED_FEATURES]
        .isna()
        .sum()
    )

    invalid_columns = missing_counts[
        missing_counts > 0
    ]

    if not invalid_columns.empty:

        raise ValueError(
            "Missing values detected in engineered features:\n"
            f"{invalid_columns}"
        )

    print(
        "Missing-value validation passed."
    )


def validate_infinite_values(
    dataframe: pd.DataFrame,
) -> None:
    """
    Ensure engineered numerical features contain no
    positive or negative infinite values.
    """

    numeric_data = (
        dataframe[ENGINEERED_FEATURES]
        .select_dtypes(include=[np.number])
    )

    infinite_counts = pd.Series(
        np.isinf(
            numeric_data.to_numpy()
        ).sum(axis=0),
        index=numeric_data.columns,
    )

    invalid_columns = infinite_counts[
        infinite_counts > 0
    ]

    if not invalid_columns.empty:

        raise ValueError(
            "Infinite values detected:\n"
            f"{invalid_columns}"
        )

    print(
        "Infinite-value validation passed."
    )


def validate_numeric_types(
    dataframe: pd.DataFrame,
) -> None:
    """
    Ensure all engineered features are numeric.
    """

    non_numeric_features = [
        feature
        for feature in ENGINEERED_FEATURES
        if not pd.api.types.is_numeric_dtype(
            dataframe[feature]
        )
    ]

    if non_numeric_features:

        raise TypeError(
            "Non-numeric engineered feature(s): "
            + ", ".join(non_numeric_features)
        )

    print(
        "Numeric data-type validation passed."
    )


def validate_feature_ranges(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate logical ranges of engineered features.
    """

    invalid_sale_month = ~dataframe[
        "sale_month"
    ].between(
        1,
        12,
    )

    if invalid_sale_month.any():

        raise ValueError(
            "sale_month contains values outside "
            "the valid range 1-12."
        )

    if (
        dataframe["house_age"] < 0
    ).any():

        raise ValueError(
            "house_age contains negative values."
        )

    if not set(
        dataframe["is_renovated"].unique()
    ).issubset({0, 1}):

        raise ValueError(
            "is_renovated must contain only 0 or 1."
        )

    if (
        dataframe[
            "years_since_renovation"
        ] < 0
    ).any():

        raise ValueError(
            "years_since_renovation contains "
            "negative values."
        )

    if (
        dataframe[
            "distance_to_city_center_km"
        ] < 0
    ).any():

        raise ValueError(
            "distance_to_city_center_km contains "
            "negative values."
        )

    print(
        "Logical feature-range validation passed."
    )


def display_feature_summary(
    dataframe: pd.DataFrame,
) -> None:
    """
    Display descriptive statistics for engineered features.
    """

    print(
        "\nEngineered Feature Summary"
    )

    print("-" * 70)

    summary = (
        dataframe[ENGINEERED_FEATURES]
        .describe()
        .transpose()
    )

    print(
        summary.to_string()
    )

    print("-" * 70)


def run_feature_validation() -> None:
    """
    Execute the complete engineered-feature validation
    pipeline.
    """

    print("=" * 70)

    print(
        "Starting Engineered Feature Validation"
    )

    print("=" * 70)

    dataframe = (
        load_engineered_dataset()
    )

    validate_required_features(
        dataframe
    )

    validate_missing_values(
        dataframe
    )

    validate_infinite_values(
        dataframe
    )

    validate_numeric_types(
        dataframe
    )

    validate_feature_ranges(
        dataframe
    )

    display_feature_summary(
        dataframe
    )

    print("=" * 70)

    print(
        "Feature validation completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":

    run_feature_validation()