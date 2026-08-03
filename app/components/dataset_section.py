"""
Dataset Overview Component.

Displays dataset statistics and preview.

Author:
    Aman Shukla
"""

from __future__ import annotations

import streamlit as st

from app.services.dataset_service import (
    load_dataset,
    get_dataset_statistics,
    get_feature_names,
)


def render_dataset_section():

    st.header("📊 Dataset Overview")

    statistics = get_dataset_statistics()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        f"{statistics['Rows']:,}"
    )

    col2.metric(
        "Columns",
        statistics["Columns"]
    )

    col3.metric(
        "Features",
        statistics["Features"]
    )

    col4.metric(
        "Target",
        statistics["Target"]
    )

    st.markdown("---")

    dataframe = load_dataset()

    st.subheader("Dataset Preview")

    st.dataframe(
        dataframe.head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    st.subheader("Available Features")

    feature_columns = get_feature_names()

    st.write(feature_columns)