"""
Project Summary Component.

Displays the project overview, technology stack,
workflow, and key project statistics.

Author:
    Aman Shukla
"""

from __future__ import annotations

import streamlit as st


def render_summary_section():

    st.header("📄 Project Overview")

    st.markdown(
        """
This project implements a **Geospatial Real Estate Valuation System**
using both traditional Machine Learning and Graph Neural Networks (GNNs).

The objective is to improve property price prediction by incorporating
spatial relationships between neighbouring properties rather than relying
only on traditional tabular features.
"""
    )

    st.markdown("---")

    # ==========================================================
    # Technology Stack
    # ==========================================================

    st.subheader("🛠 Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:

        st.markdown(
            """
### Data Processing

- Python
- Pandas
- NumPy
- GeoPandas
"""
        )

    with tech2:

        st.markdown(
            """
### Machine Learning

- XGBoost
- PyTorch
- PyTorch Geometric
- Scikit-learn
"""
        )

    with tech3:

        st.markdown(
            """
### Dashboard

- Streamlit
- Folium
- Plotly
- Matplotlib
"""
        )

    st.markdown("---")

    # ==========================================================
    # Project Workflow
    # ==========================================================

    st.subheader("🔄 Machine Learning Workflow")

    workflow = st.columns(10)

    steps = [

        "📥\nData",

        "📊\nEDA",

        "🧹\nCleaning",

        "⚙️\nFeatures",

        "🌲\nXGBoost",

        "🕸️\nKNN Graph",

        "📍\nEmbeddings",

        "🧠\nGNN",

        "📈\nEvaluation",

        "🖥️\nDashboard",

    ]

    for column, step in zip(workflow, steps):

        with column:

            st.markdown(
                f"""
    <div style="text-align:center;
    padding:12px;
    border:1px solid #d9d9d9;
    border-radius:10px;
    font-size:14px;
    font-weight:bold;">
    {step}
    </div>
    """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ==========================================================
    # Project Statistics
    # ==========================================================

    st.subheader("📌 Project Statistics")

    stat1, stat2, stat3, stat4 = st.columns(4)

    stat1.metric(
        "Dataset",
        "21,613 Houses",
    )

    stat2.metric(
        "Features",
        "27",
    )

    stat3.metric(
        "Models",
        "2",
    )

    stat4.metric(
        "Framework",
        "PyTorch Geometric",
    )


    st.markdown("---")

    st.subheader("📚 Repository Overview")

    repository_1, repository_2, repository_3 = st.columns(3)

    repository_1.metric(
    "GitHub Issues Completed",
    "17"
    )

    repository_2.metric(
    "Development Weeks",
    "4"
    )

    repository_3.metric(
    "Primary Framework",
    "PyTorch Geometric"
    )

    st.markdown("---")

    st.caption(
    "Version 1.0 • Geospatial Real Estate Valuation using Graph Neural Networks • Developed by Aman Shukla"
    )