"""
Baseline Feature Engineering Pipeline.

This module creates reusable tabular and geospatial features from the
cleaned King County Housing Dataset for baseline machine learning models.

The engineered features are designed for the Week 2 XGBoost baseline
while preserving latitude and longitude for later spatial graph
construction.

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.graph.distance import haversine_distance


# =============================================================================
# File Paths
# =============================================================================

INPUT_FILE = PROCESSED_DATA_DIR / "clean_housing_data.csv"

OUTPUT_FILE = PROCESSED_DATA_DIR / "engineered_housing_data.csv"


# =============================================================================
# Geographic Reference Point
# =============================================================================

# Approximate coordinates for central Seattle.
# This reference point is used to create a location-based distance feature.
SEATTLE_CENTER_LAT = 47.6062
SEATTLE_CENTER_LONG = -122.3321


# =============================================================================
# Required Columns
# =============================================================================

REQUIRED_COLUMNS = [
    "date",
    "price",
    "yr_built",
    "yr_renovated",
    "lat",
    "long",
]


# =============================================================================
# Dataset Loading
# =============================================================================


def load_clean_dataset() -> pd.DataFrame:
    """
    Load the cleaned King County Housing Dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned housing dataset.

    Raises
    ------
    FileNotFoundError
        If the cleaned dataset does not exist.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Cleaned housing dataset was not found.\n"
            f"Expected location: {INPUT_FILE}\n"
            "Run the preprocessing pipeline before feature engineering."
        )

    dataframe = pd.read_csv(INPUT_FILE)

    if dataframe.empty:
        raise ValueError(
            "The cleaned housing dataset is empty."
        )

    return dataframe


# =============================================================================
# Required Column Validation
# =============================================================================


def validate_required_columns(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate columns required by the feature engineering pipeline.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input housing dataset.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Feature engineering cannot continue because "
            "required columns are missing: "
            f"{missing_columns}"
        )


# =============================================================================
# Sale Date Features
# =============================================================================


def add_sale_date_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parse the property sale date and derive temporal features.

    Generated features
    ------------------
    sale_year
        Calendar year in which the property was sold.

    sale_month
        Calendar month in which the property was sold.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Housing dataset.

    Returns
    -------
    pd.DataFrame
        Dataset containing temporal features.
    """

    dataframe = dataframe.copy()

    parsed_dates = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
    )

    if parsed_dates.isna().any():
        invalid_count = int(
            parsed_dates.isna().sum()
        )

        raise ValueError(
            f"{invalid_count} invalid sale date value(s) "
            "were detected."
        )

    dataframe["sale_year"] = (
        parsed_dates.dt.year.astype("int64")
    )

    dataframe["sale_month"] = (
        parsed_dates.dt.month.astype("int64")
    )

    return dataframe


# =============================================================================
# Property Age Feature
# =============================================================================


def add_house_age_feature(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate property age at the time of sale.

    House age is calculated using the sale year rather than the current
    year to preserve temporal consistency for historical transactions.

    A small number of source records may contain a construction year
    later than the recorded sale year. These inconsistent negative ages
    are normalized to zero instead of removing the observations.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Housing dataset containing sale_year and yr_built.

    Returns
    -------
    pd.DataFrame
        Dataset containing the house_age feature.
    """

    dataframe = dataframe.copy()

    dataframe["house_age"] = (
        dataframe["sale_year"]
        - dataframe["yr_built"]
    )

    invalid_age_mask = (
        dataframe["house_age"] < 0
    )

    invalid_age_count = int(
        invalid_age_mask.sum()
    )

    if invalid_age_count > 0:

        print(
            f"Warning: {invalid_age_count} property record(s) "
            "have yr_built later than sale_year."
        )

        print(
            "Negative house-age values normalized to 0."
        )

        dataframe.loc[
            invalid_age_mask,
            "house_age"
        ] = 0

    dataframe["house_age"] = (
        dataframe["house_age"]
        .astype("int64")
    )

    return dataframe

# =============================================================================
# Renovation Features
# =============================================================================


def add_renovation_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create renovation-related features.

    Generated features
    ------------------
    is_renovated
        Binary indicator identifying whether a property has a recorded
        renovation year.

    years_since_renovation
        Number of years between renovation and sale.

        For properties with no recorded renovation, the value is set
        equal to house_age.

        If the source dataset contains a renovation year later than the
        recorded sale year, the derived years_since_renovation value is
        normalized to zero while preserving the original yr_renovated
        value for traceability.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Housing dataset containing yr_renovated, sale_year, and
        house_age.

    Returns
    -------
    pd.DataFrame
        Dataset containing renovation-related engineered features.
    """

    dataframe = dataframe.copy()

    dataframe["is_renovated"] = (
        dataframe["yr_renovated"] > 0
    ).astype("int64")

    renovated_mask = (
        dataframe["is_renovated"] == 1
    )

    invalid_renovation_mask = (
        renovated_mask
        & (
            dataframe["yr_renovated"]
            > dataframe["sale_year"]
        )
    )

    invalid_renovation_count = int(
        invalid_renovation_mask.sum()
    )

    if invalid_renovation_count > 0:

        print(
            f"Warning: {invalid_renovation_count} property record(s) "
            "have yr_renovated later than sale_year."
        )

        print(
            "Negative years-since-renovation values normalized to 0."
        )

    dataframe["years_since_renovation"] = np.where(
        renovated_mask,
        dataframe["sale_year"]
        - dataframe["yr_renovated"],
        dataframe["house_age"],
    )

    dataframe.loc[
        invalid_renovation_mask,
        "years_since_renovation"
    ] = 0

    dataframe["years_since_renovation"] = (
        dataframe["years_since_renovation"]
        .astype("int64")
    )

    return dataframe

# =============================================================================
# Distance to Seattle City Center
# =============================================================================


def add_city_center_distance(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Haversine distance from each property to central Seattle.

    This feature captures broad spatial accessibility and location
    effects that a purely structural housing model cannot represent.

    Returns
    -------
    pd.DataFrame
        Dataset containing distance_to_city_center_km.
    """

    dataframe = dataframe.copy()

    invalid_coordinates = (
        dataframe["lat"].isna()
        | dataframe["long"].isna()
        | ~dataframe["lat"].between(
            -90,
            90,
        )
        | ~dataframe["long"].between(
            -180,
            180,
        )
    )

    if invalid_coordinates.any():
        invalid_count = int(
            invalid_coordinates.sum()
        )

        raise ValueError(
            f"{invalid_count} property record(s) contain "
            "invalid geographic coordinates."
        )

    dataframe[
        "distance_to_city_center_km"
    ] = [
        haversine_distance(
            latitude_1=latitude,
            longitude_1=longitude,
            latitude_2=SEATTLE_CENTER_LAT,
            longitude_2=SEATTLE_CENTER_LONG,
        )
        for latitude, longitude in zip(
            dataframe["lat"],
            dataframe["long"],
        )
    ]

    return dataframe


# =============================================================================
# Engineered Feature Validation
# =============================================================================


def validate_engineered_features(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate all features generated by this module.
    """

    engineered_columns = [
        "sale_year",
        "sale_month",
        "house_age",
        "is_renovated",
        "years_since_renovation",
        "distance_to_city_center_km",
    ]

    missing_columns = [
        column
        for column in engineered_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Engineered features are missing: "
            f"{missing_columns}"
        )

    if (
        dataframe[engineered_columns]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Missing values were detected in "
            "the engineered features."
        )

    if not dataframe["sale_month"].between(
        1,
        12,
    ).all():
        raise ValueError(
            "Invalid values detected in sale_month."
        )

    if (
        dataframe["house_age"] < 0
    ).any():
        raise ValueError(
            "Negative house_age values detected."
        )

    if not dataframe["is_renovated"].isin(
        [0, 1]
    ).all():
        raise ValueError(
            "is_renovated must contain only 0 or 1."
        )

    if (
        dataframe["years_since_renovation"] < 0
    ).any():
        raise ValueError(
            "Negative years_since_renovation values detected."
        )

    if (
        dataframe["distance_to_city_center_km"] < 0
    ).any():
        raise ValueError(
            "Negative geographic distances detected."
        )


# =============================================================================
# Main Feature Engineering Pipeline
# =============================================================================


def engineer_features() -> pd.DataFrame:
    """
    Execute the complete baseline feature engineering pipeline.

    Returns
    -------
    pd.DataFrame
        Feature-engineered housing dataset.
    """

    print("=" * 70)
    print("Starting Baseline Feature Engineering Pipeline")
    print("=" * 70)

    dataframe = load_clean_dataset()

    print(
        f"Clean dataset loaded: "
        f"{len(dataframe):,} rows, "
        f"{len(dataframe.columns)} columns."
    )

    validate_required_columns(
        dataframe
    )

    print(
        "Required column validation passed."
    )

    dataframe = add_sale_date_features(
        dataframe
    )

    print(
        "Sale date features generated."
    )

    dataframe = add_house_age_feature(
        dataframe
    )

    print(
        "House age feature generated."
    )

    dataframe = add_renovation_features(
        dataframe
    )

    print(
        "Renovation features generated."
    )

    dataframe = add_city_center_distance(
        dataframe
    )

    print(
        "Distance-to-city-center feature generated."
    )

    validate_engineered_features(
        dataframe
    )

    print(
        "Engineered feature validation passed."
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "Feature-engineered dataset saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Generated Features:"
    )

    generated_features = [
        "sale_year",
        "sale_month",
        "house_age",
        "is_renovated",
        "years_since_renovation",
        "distance_to_city_center_km",
    ]

    for feature in generated_features:
        print(
            f"- {feature}"
        )

    print()
    print(
        f"Final dataset shape: "
        f"{dataframe.shape}"
    )

    print("=" * 70)
    print(
        "Feature engineering completed successfully."
    )
    print("=" * 70)

    return dataframe


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    engineer_features()