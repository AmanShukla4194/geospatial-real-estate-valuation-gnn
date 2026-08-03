"""
Model Performance Component.

Displays model evaluation metrics and comparison.

Author:
    Aman Shukla
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.metrics_service import (
    get_baseline_metrics,
    get_gnn_metrics,
)


def render_metrics_section():

    st.header("📈 Model Performance")

    baseline = get_baseline_metrics()

    gnn = get_gnn_metrics()

    # ==========================================================
    # KPI Cards
    # ==========================================================

    st.subheader("Model Comparison")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### XGBoost Baseline")

        st.metric(
            "MAE",
            f"{baseline['MAE']:,.2f}"
        )

        st.metric(
            "RMSE",
            f"{baseline['RMSE']:,.2f}"
        )

        st.metric(
            "MAPE",
            f"{baseline['MAPE']:.2f}%"
        )

        st.metric(
            "R²",
            f"{baseline['R²']:.4f}"
        )

    with col2:

        st.markdown("### Graph Neural Network")

        st.metric(
            "MAE",
            f"{gnn['MAE']:,.2f}"
        )

        st.metric(
            "RMSE",
            f"{gnn['RMSE']:,.2f}"
        )

        st.metric(
            "MAPE",
            f"{gnn['MAPE']:.2f}%"
        )

        st.metric(
            "R²",
            f"{gnn['R²']:.4f}"
        )

    st.markdown("---")

    # ==========================================================
    # Comparison Table
    # ==========================================================

    st.subheader("Evaluation Summary")

    comparison = pd.DataFrame(

        [

            baseline,

            gnn,

        ]

    )

    st.dataframe(

        comparison,

        use_container_width=True,

        hide_index=True,

    )