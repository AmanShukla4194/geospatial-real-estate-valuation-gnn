"""
Exploratory Data Analysis (EDA) Pipeline.

This module performs a complete exploratory data analysis of the
King County Housing Dataset.

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from pathlib import Path

from src.config import REPORTS_DIR
from src.preprocessing.load_dataset import (
    load_dataset,
    validate_columns
)

from src.utils.statistics import (
    dataset_shape,
    descriptive_statistics,
    missing_value_summary,
    correlation_matrix,
    unique_value_summary,
    dataframe_to_markdown
)

from src.utils.visualization import (
    plot_histogram,
    plot_boxplot,
    plot_correlation_heatmap
)


REPORT_FILE = REPORTS_DIR / "eda_summary.md"


def generate_markdown_report(
    shape_info,
    missing_summary,
    statistics_summary,
    unique_summary
):
    """
    Generate a Markdown EDA report.
    """

    with open(REPORT_FILE, "w", encoding="utf-8") as report:

        report.write("# Exploratory Data Analysis Report\n\n")

        report.write("## Dataset Overview\n\n")

        report.write(f"- Rows: {shape_info['rows']}\n")
        report.write(f"- Columns: {shape_info['columns']}\n\n")

        report.write("## Missing Values\n\n")

        report.write(
            dataframe_to_markdown(missing_summary)
        )

        report.write("\n\n")

        report.write("## Descriptive Statistics\n\n")

        report.write(
            dataframe_to_markdown(statistics_summary)
        )

        report.write("\n\n")

        report.write("## Unique Value Summary\n\n")

        report.write(
            dataframe_to_markdown(unique_summary)
        )


def run_eda():

    print("=" * 70)
    print("Starting Exploratory Data Analysis")
    print("=" * 70)

    dataframe = load_dataset()

    validate_columns(dataframe)

    print("Dataset loaded successfully.\n")

    shape_info = dataset_shape(dataframe)

    missing_summary = missing_value_summary(dataframe)

    statistics_summary = descriptive_statistics(dataframe)

    unique_summary = unique_value_summary(dataframe)

    corr_matrix = correlation_matrix(dataframe)

    print("Generating visualizations...")

    plot_histogram(dataframe, "price")

    plot_histogram(dataframe, "bedrooms")

    plot_histogram(dataframe, "bathrooms")

    plot_histogram(dataframe, "sqft_living")

    plot_boxplot(dataframe, "price")

    plot_boxplot(dataframe, "sqft_living")

    plot_correlation_heatmap(corr_matrix)

    print("Generating Markdown report...")

    generate_markdown_report(
        shape_info,
        missing_summary,
        statistics_summary,
        unique_summary
    )

    print("\nEDA completed successfully.")

    print(f"\nReport saved to:\n{REPORT_FILE}")

    print("=" * 70)


if __name__ == "__main__":
    run_eda()