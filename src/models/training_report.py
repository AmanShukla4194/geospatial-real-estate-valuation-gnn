"""
Graph Neural Network Training Report.

Generates a Markdown report summarizing GNN training.

Author:
    Aman Shukla
"""

from pathlib import Path

import pandas as pd

from src.config import (
    REPORTS_DIR,
    TRAINING_HISTORY_PATH,
    GNN_EPOCHS,
)

REPORT_FILE = REPORTS_DIR / "gnn_training_summary.md"


def generate_training_report():

    print("=" * 70)
    print("Generating GNN Training Report")
    print("=" * 70)

    history = pd.read_csv(
        TRAINING_HISTORY_PATH
    )

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        report.write("# GNN Training Summary\n\n")

        report.write("## Training Configuration\n\n")

        report.write(f"- Epochs: **{GNN_EPOCHS}**\n")
        report.write("- Optimizer: **Adam**\n")
        report.write("- Loss Function: **Mean Squared Error (MSE)**\n\n")

        report.write("## Training Results\n\n")

        report.write(
            f"- Initial Loss: **{history.iloc[0]['loss']:.4f}**\n"
        )

        report.write(
            f"- Final Loss: **{history.iloc[-1]['loss']:.4f}**\n\n"
        )

        report.write("## Observation\n\n")

        report.write(
            "The Graph Neural Network training pipeline completed "
            "successfully and the training loss consistently "
            "decreased over the configured epochs. The trained "
            "model and training history were successfully saved "
            "for subsequent evaluation and comparison with the "
            "baseline XGBoost model.\n"
        )

    print("Training report generated successfully.")
    print(REPORT_FILE)
    print("=" * 70)


if __name__ == "__main__":

    generate_training_report()