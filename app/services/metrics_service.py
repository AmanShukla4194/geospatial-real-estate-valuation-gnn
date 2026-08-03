"""
Metrics Service.

Provides model evaluation metrics for the dashboard.

Author:
    Aman Shukla
"""

from __future__ import annotations


def get_baseline_metrics():

    return {

        "Model": "XGBoost",

        "MAE": 76713.21,

        "RMSE": 105325.82,

        "MAPE": 17.19,

        "R²": 0.8271,
    }


def get_gnn_metrics():

    return {

        "Model": "Graph Neural Network",

        "MAE": 399649.94,

        "RMSE": 467054.00,

        "MAPE": 73.51,

        "R²": -2.5972,
    }