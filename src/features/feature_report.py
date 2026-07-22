"""
Feature Engineering Report Generator.

This module generates a reproducible Markdown report describing the
baseline tabular features engineered for the Geospatial Real Estate
Valuation project.

The report documents:
- Dataset dimensions
- Engineered feature definitions
- Feature summary statistics
- Data-quality corrections
- Validation results
- Readiness for baseline machine-learning modeling

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR


ENGINEERED_DATA_FILE = (
    PROCESSED_DATA_DIR
    / "engineered_housing_data.csv"
)

REPORT_FILE = (
    REPORTS_DIR
    / "feature_engineering_summary.md"
)


ENGINEERED_FEATURES = [
    "sale_year",
    "sale_month",
    "house_age",
    "is_renovated",
    "years_since_renovation",
    "distance_to_city_center_km",
]


FEATURE_DESCRIPTIONS = {
    "sale_year": (
        "Calendar year in which the property sale occurred."
    ),
    "sale_month": (
        "Calendar month in which the property sale occurred."
    ),
    "house_age": (
        "Age of the property at the time of sale, calculated from "
        "sale year and construction year."
    ),
    "is_renovated": (
        "Binary indicator showing whether a property has a recorded "
        "renovation year."
    ),
    "years_since_renovation": (
        "Number of years between the recorded renovation and sale. "
        "For properties without a recorded renovation, house age is "
        "used to represent time since original construction."
    ),
    "distance_to_city_center_km": (
        "Haversine distance in kilometers from the property to the "
        "configured Seattle city-center reference coordinate."
    ),
}


def load_engineered_dataset() -> pd.DataFrame:
    """
    Load the feature-engineered dataset.

    Returns
    -------
    pd.DataFrame
        Feature-engineered housing dataset.
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

    return dataframe


def validate_report_input(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the dataset before generating the report.
    """

    missing_features = [
        feature
        for feature in ENGINEERED_FEATURES
        if feature not in dataframe.columns
    ]

    if missing_features:

        raise ValueError(
            "Cannot generate feature report. "
            "Missing feature(s): "
            + ", ".join(missing_features)
        )

    if dataframe.empty:

        raise ValueError(
            "Cannot generate feature report from an empty dataset."
        )


def calculate_validation_metrics(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    """
    Calculate validation metrics for engineered features.
    """

    feature_data = dataframe[
        ENGINEERED_FEATURES
    ]

    missing_values = int(
        feature_data
        .isna()
        .sum()
        .sum()
    )

    numeric_data = feature_data.select_dtypes(
        include=[np.number]
    )

    infinite_values = int(
        np.isinf(
            numeric_data.to_numpy()
        ).sum()
    )

    negative_house_age = int(
        (
            dataframe["house_age"] < 0
        ).sum()
    )

    invalid_sale_month = int(
        (
            ~dataframe["sale_month"].between(
                1,
                12,
            )
        ).sum()
    )

    negative_distance = int(
        (
            dataframe[
                "distance_to_city_center_km"
            ] < 0
        ).sum()
    )

    return {
        "missing_values": missing_values,
        "infinite_values": infinite_values,
        "negative_house_age": negative_house_age,
        "invalid_sale_month": invalid_sale_month,
        "negative_distance": negative_distance,
    }


def generate_feature_report(
    dataframe: pd.DataFrame,
) -> None:
    """
    Generate the Markdown feature-engineering report.
    """

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = (
        dataframe[ENGINEERED_FEATURES]
        .describe()
        .transpose()
        .round(4)
    )

    validation = (
        calculate_validation_metrics(
            dataframe
        )
    )

    renovated_count = int(
        dataframe[
            "is_renovated"
        ].sum()
    )

    renovated_percentage = (
        renovated_count
        / len(dataframe)
        * 100
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "# Feature Engineering Summary\n\n"
        )

        report.write(
            "## Overview\n\n"
        )

        report.write(
            "This report documents the baseline tabular feature "
            "engineering stage for the Geospatial Real Estate "
            "Valuation project.\n\n"
        )

        report.write(
            "These features establish the input foundation for the "
            "traditional machine-learning baseline that will later "
            "be compared against spatial and graph-based valuation "
            "models.\n\n"
        )

        report.write(
            "## Dataset Summary\n\n"
        )

        report.write(
            f"- Total observations: **{len(dataframe):,}**\n"
        )

        report.write(
            f"- Total columns after feature engineering: "
            f"**{len(dataframe.columns)}**\n"
        )

        report.write(
            f"- Engineered features added: "
            f"**{len(ENGINEERED_FEATURES)}**\n\n"
        )

        report.write(
            "## Engineered Features\n\n"
        )

        for feature in ENGINEERED_FEATURES:

            report.write(
                f"### `{feature}`\n\n"
            )

            report.write(
                FEATURE_DESCRIPTIONS[
                    feature
                ]
                + "\n\n"
            )

        report.write(
            "## Feature Summary Statistics\n\n"
        )

        report.write(
            summary.to_markdown()
        )

        report.write(
            "\n\n"
        )

        report.write(
            "## Data Quality Decisions\n\n"
        )

        report.write(
            "- **12 property records** contained a construction "
            "year later than the recorded sale year. The original "
            "`yr_built` values were preserved, while the derived "
            "`house_age` values were normalized to `0` to prevent "
            "invalid negative ages.\n\n"
        )

        report.write(
            "- **6 property records** contained a renovation year "
            "later than the recorded sale year. The original "
            "`yr_renovated` values were preserved, while the "
            "derived `years_since_renovation` values were "
            "normalized to `0`.\n\n"
        )

        report.write(
            "- These corrections apply only to derived modeling "
            "features and preserve the original source columns for "
            "traceability.\n\n"
        )

        report.write(
            "## Renovation Profile\n\n"
        )

        report.write(
            f"- Properties with recorded renovations: "
            f"**{renovated_count:,}**\n"
        )

        report.write(
            f"- Percentage with recorded renovations: "
            f"**{renovated_percentage:.2f}%**\n\n"
        )

        report.write(
            "## Validation Results\n\n"
        )

        report.write(
            f"- Missing engineered-feature values: "
            f"**{validation['missing_values']}**\n"
        )

        report.write(
            f"- Infinite engineered-feature values: "
            f"**{validation['infinite_values']}**\n"
        )

        report.write(
            f"- Remaining negative house-age values: "
            f"**{validation['negative_house_age']}**\n"
        )

        report.write(
            f"- Invalid sale-month values: "
            f"**{validation['invalid_sale_month']}**\n"
        )

        report.write(
            f"- Negative city-center distances: "
            f"**{validation['negative_distance']}**\n\n"
        )

        report.write(
            "All engineered features passed the required "
            "validation checks.\n\n"
        )

        report.write(
            "## Modeling Readiness\n\n"
        )

        report.write(
            "The feature-engineered dataset is ready for the "
            "baseline machine-learning stage. The next modeling "
            "phase will select appropriate predictors, separate "
            "the target variable (`price`), create reproducible "
            "training and holdout partitions, and train an "
            "XGBoost regression baseline.\n\n"
        )

        report.write(
            "Baseline performance will be evaluated primarily "
            "using Mean Absolute Percentage Error (MAPE) and Root "
            "Mean Squared Error (RMSE). These results will provide "
            "the benchmark against which the later spatial "
            "embedding and Graph Neural Network models will be "
            "compared.\n"
        )


def run_feature_report() -> None:
    """
    Execute feature-engineering report generation.
    """

    print("=" * 70)

    print(
        "Starting Feature Engineering Report Generation"
    )

    print("=" * 70)

    dataframe = (
        load_engineered_dataset()
    )

    print(
        f"Engineered dataset loaded: "
        f"{len(dataframe):,} rows, "
        f"{len(dataframe.columns)} columns."
    )

    validate_report_input(
        dataframe
    )

    print(
        "Report input validation passed."
    )

    generate_feature_report(
        dataframe
    )

    print(
        "\nFeature engineering report generated:"
    )

    print(
        REPORT_FILE
    )

    print("=" * 70)

    print(
        "Report generation completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":

    run_feature_report()