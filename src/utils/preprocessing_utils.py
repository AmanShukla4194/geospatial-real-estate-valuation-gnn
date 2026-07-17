"""
Preprocessing utility functions.

This module provides reusable preprocessing functions used throughout
the Geospatial Real Estate Valuation project.

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from pathlib import Path

import pandas as pd
import numpy as np


# =============================================================================
# Duplicate Records
# =============================================================================

def remove_duplicates(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows from the dataset.

    Returns
    -------
    tuple
        Clean dataframe and number of duplicates removed.
    """

    before = len(dataframe)

    dataframe = dataframe.drop_duplicates()

    after = len(dataframe)

    removed = before - after

    return dataframe, removed


# =============================================================================
# Missing Values
# =============================================================================

def missing_value_summary(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate missing value summary.
    """

    summary = pd.DataFrame({
        "Missing Values": dataframe.isnull().sum(),
        "Percentage": (
            dataframe.isnull().sum()
            / len(dataframe)
            * 100
        ).round(2)
    })

    return summary


def fill_missing_values(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill missing values.

    Numerical columns → Median

    Categorical columns → Mode
    """

    dataframe = dataframe.copy()

    numeric_columns = dataframe.select_dtypes(
        include=np.number
    ).columns

    categorical_columns = dataframe.select_dtypes(
        exclude=np.number
    ).columns

    for column in numeric_columns:

        dataframe[column] = dataframe[column].fillna(
            dataframe[column].median()
        )

    for column in categorical_columns:

        dataframe[column] = dataframe[column].fillna(
            dataframe[column].mode()[0]
        )

    return dataframe


# =============================================================================
# Numeric Validation
# =============================================================================

def validate_numeric_columns(
    dataframe: pd.DataFrame,
    columns: list[str]
) -> None:
    """
    Ensure specified columns are numeric.
    """

    for column in columns:

        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        ):

            raise TypeError(
                f"{column} must be numeric."
            )


# =============================================================================
# Outlier Handling
# =============================================================================

def cap_outliers_iqr(
    dataframe: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """
    Cap extreme outliers using the IQR method.
    """

    dataframe = dataframe.copy()

    q1 = dataframe[column].quantile(0.25)

    q3 = dataframe[column].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - (1.5 * iqr)

    upper = q3 + (1.5 * iqr)

    dataframe[column] = dataframe[column].clip(
        lower=lower,
        upper=upper
    )

    return dataframe


# =============================================================================
# Dataset Saving
# =============================================================================

def save_clean_dataset(
    dataframe: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save cleaned dataset.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe.to_csv(
        output_path,
        index=False
    )