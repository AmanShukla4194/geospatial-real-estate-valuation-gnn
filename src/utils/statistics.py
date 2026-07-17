"""
Statistical utility functions for Exploratory Data Analysis (EDA).

This module provides reusable statistical functions used throughout the
Geospatial Real Estate Valuation project.

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


def dataset_shape(dataframe: pd.DataFrame) -> Dict[str, int]:
    """
    Return the dataset dimensions.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    Returns
    -------
    Dict[str, int]
        Dictionary containing row and column counts.
    """

    return {
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1]
    }


def missing_value_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate missing value statistics.

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Missing value summary.
    """

    summary = pd.DataFrame({
        "Missing Values": dataframe.isnull().sum(),
        "Percentage": (
            dataframe.isnull().sum()
            / len(dataframe)
            * 100
        ).round(2)
    })

    return summary.sort_values(
        by="Missing Values",
        ascending=False
    )


def descriptive_statistics(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate descriptive statistics.

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    return dataframe.describe().transpose()


def correlation_matrix(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix.

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    numeric_dataframe = dataframe.select_dtypes(
        include=["number"]
    )

    return numeric_dataframe.corr()


def unique_value_summary(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Count unique values for each feature.

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    return pd.DataFrame({
        "Unique Values": dataframe.nunique()
    }).sort_values(
        by="Unique Values",
        ascending=False
    )

def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """
    Convert a pandas DataFrame into a Markdown table without requiring
    the external 'tabulate' package.

    Parameters
    ----------
    dataframe : pd.DataFrame

    Returns
    -------
    str
    """

    headers = list(dataframe.columns)

    markdown = []

    markdown.append("| Index | " + " | ".join(headers) + " |")

    markdown.append("|" + "---|" * (len(headers) + 1))

    for index, row in dataframe.iterrows():

        values = [str(value) for value in row.tolist()]

        markdown.append(
            f"| {index} | " + " | ".join(values) + " |"
        )

    return "\n".join(markdown)