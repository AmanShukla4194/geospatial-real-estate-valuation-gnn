"""
Project Information Component.

Displays project metadata, repository statistics,
technology stack, and development overview.

Author:
    Aman Shukla
"""

from __future__ import annotations

import streamlit as st

from src.project_info import (
    PROJECT_NAME,
    PROJECT_SUBTITLE,
    PROJECT_DESCRIPTION,
    DATASET_NAME,
    DATASET_ROWS,
    DATASET_COLUMNS,
    BASELINE_MODEL,
    GRAPH_MODEL,
    FRAMEWORK,
    VERSION,
    AUTHOR,
)


def render_project_information():

    st.header("ℹ️ Project Information")

    st.markdown(PROJECT_DESCRIPTION)

    st.markdown("---")

    st.subheader("Dataset")

    st.info(
        f"{DATASET_NAME}\n\n"
        f"Rows: {DATASET_ROWS:,}\n\n"
        f"Features: {DATASET_COLUMNS}"
    )

    st.subheader("Models")

    col1, col2 = st.columns(2)

    with col1:

        st.success(BASELINE_MODEL)

    with col2:

        st.success(GRAPH_MODEL)

    st.markdown("---")

    st.subheader("Technology Stack")

    st.write("• Python")

    st.write("• Pandas")

    st.write("• GeoPandas")

    st.write("• Scikit-learn")

    st.write("• XGBoost")

    st.write("• PyTorch")

    st.write("• PyTorch Geometric")

    st.write("• Streamlit")

    st.write("• Folium")

    st.write("• Plotly")

    st.markdown("---")

    st.subheader("Repository")

    repo1, repo2, repo3 = st.columns(3)

    repo1.metric(
        "Version",
        VERSION,
    )

    repo2.metric(
        "Framework",
        FRAMEWORK,
    )

    repo3.metric(
        "Developer",
        AUTHOR,
    )