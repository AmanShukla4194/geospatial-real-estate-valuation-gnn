"""
Baseline Model Report Generator.

Generates a Markdown report summarizing the baseline XGBoost model
training and evaluation results.

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from __future__ import annotations

from pathlib import Path

from src.training.model_evaluation import evaluate_model
from src.config import REPORTS_DIR

REPORT_FILE = REPORTS_DIR / "baseline_model_summary.md"


def generate_report() -> None:
    """
    Generate baseline model evaluation report.
    """

    metrics = evaluate_model()

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        report.write("# Baseline XGBoost Model Report\n\n")

        report.write("## Overview\n\n")

        report.write(
            "This report summarizes the baseline XGBoost regression "
            "model developed during Week 2 of the project. "
            "The model serves as the benchmark for comparison with "
            "the future Graph Neural Network (GNN) model.\n\n"
        )

        report.write("## Evaluation Metrics\n\n")

        report.write(f"- **MAE:** {metrics['MAE']:,.2f}\n")
        report.write(f"- **RMSE:** {metrics['RMSE']:,.2f}\n")
        report.write(f"- **MAPE:** {metrics['MAPE']:.2f}%\n")
        report.write(f"- **R² Score:** {metrics['R2']:.4f}\n\n")

        report.write("## Interpretation\n\n")

        report.write(
            f"- The baseline model explains approximately "
            f"**{metrics['R2'] * 100:.2f}%** of the variance "
            "in housing prices.\n"
        )

        report.write(
            f"- The current benchmark MAPE is "
            f"**{metrics['MAPE']:.2f}%**.\n"
        )

        report.write(
            "- The objective of the spatial graph model is to "
            "reduce this MAPE by incorporating neighborhood "
            "relationships and spatial embeddings.\n\n"
        )

        report.write("## Next Phase\n\n")

        report.write(
            "Week 3 will construct a K-Nearest Neighbor (KNN) graph "
            "and prepare spatial embeddings for Graph Neural Network "
            "training. The GNN will be evaluated against this "
            "baseline using the same regression metrics.\n"
        )

    print("=" * 70)
    print("Baseline report generated successfully.")
    print(REPORT_FILE)
    print("=" * 70)


if __name__ == "__main__":
    generate_report()