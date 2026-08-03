"""
Model Comparison Report Generator.

Generates a Markdown report and visualization comparing
the XGBoost baseline with the Graph Neural Network.

Author:
    Aman Shukla
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.config import (
    MODEL_COMPARISON_REPORT,
    MODEL_COMPARISON_FIGURE,
)

from src.evaluation.compare_models import (
    compare_models,
)


def generate_report():

    print("=" * 70)
    print("Generating Model Comparison Report")
    print("=" * 70)

    comparison = compare_models()

    MODEL_COMPARISON_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_COMPARISON_FIGURE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==========================================================
    # Markdown Report
    # ==========================================================

    with open(
        MODEL_COMPARISON_REPORT,
        "w",
        encoding="utf-8",
    ) as report:

        report.write("# Model Comparison Summary\n\n")

        report.write("## Evaluation Metrics\n\n")

        report.write(
            comparison.to_markdown(index=False)
        )

        report.write("\n\n")

        report.write("## Observations\n\n")

        report.write(
            "- XGBoost currently outperforms the Graph Neural Network "
            "on the selected evaluation metrics.\n"
        )

        report.write(
            "- The Graph Neural Network successfully completed "
            "training and inference.\n"
        )

        report.write(
            "- Additional hyperparameter tuning, graph refinement, "
            "and feature engineering may improve GNN performance.\n"
        )

        report.write(
            "- The complete graph-based pipeline has been implemented "
            "successfully.\n"
        )

    # ==========================================================
    # Visualization
    # ==========================================================

    metrics = [
        "MAE",
        "RMSE",
        "MAPE",
        "R2",
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
    )

    axes = axes.flatten()

    for axis, metric in zip(
        axes,
        metrics,
    ):

        axis.bar(
            comparison["Model"],
            comparison[metric],
        )

        axis.set_title(metric)

        axis.tick_params(
            axis="x",
            rotation=15,
        )

    plt.tight_layout()

    plt.savefig(
        MODEL_COMPARISON_FIGURE,
        dpi=300,
    )

    plt.close()

    print()

    print(
        "Markdown report saved:"
    )

    print(
        MODEL_COMPARISON_REPORT
    )

    print()

    print(
        "Comparison figure saved:"
    )

    print(
        MODEL_COMPARISON_FIGURE
    )

    print()

    print("=" * 70)


if __name__ == "__main__":

    generate_report()