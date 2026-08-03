"""
Model Comparison Module.

Compares the baseline XGBoost model with the Graph Neural Network.

Author:
    Aman Shukla
"""

from __future__ import annotations

import pandas as pd

from src.models.evaluate_gnn import (
    evaluate_gnn,
)

# =============================================================================
# Baseline Metrics
# =============================================================================

# Week 2 baseline evaluation results

BASELINE_RESULTS = {

    "Model": "XGBoost",

    "MAE": 76713.21,

    "RMSE": 105325.82,

    "MAPE": 17.19,

    "R2": 0.8271,
}


# =============================================================================
# Comparison
# =============================================================================

def compare_models():

    print("=" * 70)

    print(
        "Comparing XGBoost and Graph Neural Network"
    )

    print("=" * 70)

    gnn = evaluate_gnn()

    comparison = pd.DataFrame(

        [

            BASELINE_RESULTS,

            {

                "Model": "Graph Neural Network",

                "MAE": gnn["MAE"],

                "RMSE": gnn["RMSE"],

                "MAPE": gnn["MAPE"],

                "R2": gnn["R2"],
            },

        ]

    )

    print()

    print(comparison)

    print()

    print("=" * 70)

    return comparison


if __name__ == "__main__":

    compare_models()