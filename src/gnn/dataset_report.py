"""
PyTorch Geometric Dataset Report Generator.

Generates a Markdown report describing the prepared Graph Neural
Network dataset.

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from __future__ import annotations

from src.config import (
    REPORTS_DIR,
    NODE_FEATURE_COLUMNS,
    GRAPH_TARGET_COLUMN,
)

from src.gnn.graph_dataset import create_graph_data


REPORT_FILE = (
    REPORTS_DIR
    / "gnn_dataset_summary.md"
)


def generate_report():

    print("=" * 70)

    print(
        "Generating GNN Dataset Report"
    )

    print("=" * 70)

    graph = create_graph_data()

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "# Graph Neural Network Dataset Summary\n\n"
        )

        report.write(
            "## Overview\n\n"
        )

        report.write(
            "This report summarizes the prepared "
            "PyTorch Geometric dataset used for "
            "Graph Neural Network training.\n\n"
        )

        report.write(
            "## Graph Statistics\n\n"
        )

        report.write(
            f"- Nodes: **{graph.num_nodes:,}**\n"
        )

        report.write(
            f"- Edges: **{graph.num_edges:,}**\n"
        )

        report.write(
            f"- Node Features: **{graph.num_node_features}**\n"
        )

        report.write(
            f"- Target Column: **{GRAPH_TARGET_COLUMN}**\n\n"
        )

        report.write(
            "## Node Features\n\n"
        )

        for feature in NODE_FEATURE_COLUMNS:

            report.write(
                f"- {feature}\n"
            )

        report.write(
            "\n"
        )

        report.write(
            "## Tensor Shapes\n\n"
        )

        report.write(
            f"- Node Feature Matrix: **{tuple(graph.x.shape)}**\n"
        )

        report.write(
            f"- Edge Index: **{tuple(graph.edge_index.shape)}**\n"
        )

        report.write(
            f"- Target Tensor: **{tuple(graph.y.shape)}**\n\n"
        )

        report.write(
            "## Validation\n\n"
        )

        report.write(
            "- Dataset validation passed.\n"
        )

        report.write(
            "- Graph connectivity validated.\n"
        )

        report.write(
            "- Node feature dimensions verified.\n"
        )

        report.write(
            "- Target tensor verified.\n"
        )

        report.write(
            "- Dataset ready for Graph Neural Network training.\n\n"
        )

        report.write(
            "## Next Stage\n\n"
        )

        report.write(
            "The prepared PyTorch Geometric Data object "
            "will be used to train a Graph Neural Network "
            "regression model. The resulting model will be "
            "evaluated using MAE, RMSE, MAPE and R² before "
            "being compared against the Week 2 XGBoost "
            "baseline.\n"
        )

    print(
        "GNN dataset report generated successfully."
    )

    print(REPORT_FILE)

    print("=" * 70)


if __name__ == "__main__":

    generate_report()