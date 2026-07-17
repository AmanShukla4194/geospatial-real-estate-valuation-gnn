"""
Data Cleaning Pipeline.

This module performs complete preprocessing of the King County Housing
Dataset before feature engineering and model training.

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from src.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    OUTLIER_COLUMNS,
    NUMERIC_VALIDATION_COLUMNS,
)

from src.preprocessing.load_dataset import (
    load_dataset,
    validate_columns,
)

from src.utils.preprocessing_utils import (
    remove_duplicates,
    missing_value_summary,
    fill_missing_values,
    validate_numeric_columns,
    cap_outliers_iqr,
    save_clean_dataset,
)

# =============================================================================
# Output Files
# =============================================================================

OUTPUT_FILE = PROCESSED_DATA_DIR / "clean_housing_data.csv"

REPORT_FILE = REPORTS_DIR / "preprocessing_summary.md"


# =============================================================================
# Report Generation
# =============================================================================

def generate_preprocessing_report(
    duplicate_count,
    missing_summary,
):
    """
    Generate preprocessing report in Markdown format.
    """

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as report:

        report.write("# Data Preprocessing Report\n\n")

        report.write("## Duplicate Records\n\n")

        report.write(
            f"- Duplicate rows removed: **{duplicate_count}**\n\n"
        )

        report.write("## Missing Values (Before Cleaning)\n\n")

        report.write("```\n")
        report.write(missing_summary.to_string())
        report.write("\n```\n\n")

        report.write("## Processing Steps\n\n")

        report.write("- Loaded raw housing dataset\n")
        report.write("- Removed duplicate records\n")
        report.write("- Filled missing numerical values using median\n")
        report.write("- Filled missing categorical values using mode\n")
        report.write("- Validated required numerical columns\n")
        report.write("- Normalized numerical outliers using the IQR method\n")
        report.write("- Exported cleaned dataset\n\n")

        report.write("## Deliverables\n\n")

        report.write("- Clean dataset generated successfully\n")
        report.write("- Missing value summary generated\n")
        report.write("- Duplicate removal completed\n")
        report.write("- Numeric validation completed\n")
        report.write("- Outlier normalization completed\n")
        report.write("- Dataset ready for feature engineering\n")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_preprocessing():
    """
    Execute complete preprocessing pipeline.
    """

    print("=" * 70)
    print("Starting Data Cleaning Pipeline")
    print("=" * 70)

    dataframe = load_dataset()

    validate_columns(dataframe)

    print("Dataset loaded.")

    dataframe, removed = remove_duplicates(dataframe)

    print(f"Duplicate rows removed : {removed}")

    summary = missing_value_summary(dataframe)

    print("\nMissing Value Summary")
    print(summary)

    dataframe = fill_missing_values(dataframe)

    validate_numeric_columns(
        dataframe,
        NUMERIC_VALIDATION_COLUMNS
    )

    print("\nNumeric column validation passed.")

    print("\nNormalizing outliers...")

    for column in OUTLIER_COLUMNS:

        dataframe = cap_outliers_iqr(
            dataframe,
            column
        )

    save_clean_dataset(
        dataframe,
        OUTPUT_FILE
    )

    generate_preprocessing_report(
        removed,
        summary,
    )

    print("\nClean dataset saved.")
    print(OUTPUT_FILE)

    print("\nPreprocessing report generated.")
    print(REPORT_FILE)

    print("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run_preprocessing()