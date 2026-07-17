"""
Dataset loading and validation module.

This module loads the King County housing dataset, validates its
structure, and provides reusable utilities for downstream processing.

Author:
    Aman Shukla
"""

from pathlib import Path

import pandas as pd

from src.config import (
    DATASET_PATH,
    REQUIRED_COLUMNS
)


def validate_dataset_path(dataset_path: Path) -> None:
    """
    Validate that the dataset exists.

    Parameters
    ----------
    dataset_path : Path

    Raises
    ------
    FileNotFoundError
    """

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_path}"
        )


def load_dataset() -> pd.DataFrame:
    """
    Load the housing dataset.

    Returns
    -------
    pandas.DataFrame
    """

    validate_dataset_path(DATASET_PATH)

    dataframe = pd.read_csv(DATASET_PATH)

    return dataframe


def validate_columns(dataframe: pd.DataFrame) -> None:
    """
    Validate required columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame

    Raises
    ------
    ValueError
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns:\n"
            + "\n".join(missing_columns)
        )


def print_dataset_summary(dataframe: pd.DataFrame) -> None:
    """
    Print dataset information.
    """

    print("=" * 60)
    print("KING COUNTY HOUSING DATASET")
    print("=" * 60)

    print(f"Rows: {dataframe.shape[0]}")
    print(f"Columns: {dataframe.shape[1]}")

    print("\nColumns:")

    for column in dataframe.columns:
        print(f"• {column}")

    print("\nMissing Values")

    print(dataframe.isnull().sum())

    print("\nData Types")

    print(dataframe.dtypes)

    print("=" * 60)


def main() -> None:

    dataframe = load_dataset()

    validate_columns(dataframe)

    print_dataset_summary(dataframe)


if __name__ == "__main__":
    main()