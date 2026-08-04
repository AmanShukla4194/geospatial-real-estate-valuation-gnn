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

    # ==========================================================
    # Project Overview
    # ==========================================================

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
    # Machine Learning Workflow
    # ==========================================================

    st.subheader("🔄 Machine Learning Workflow")

    workflow = [
        "📥\nData",
        "📊\nEDA",
        "🧹\nCleaning",
        "⚙️\nFeatures",
        "🌲\nXGBoost",
        "🕸️\nKNN\nGraph",
        "📍\nEmbeddings",
        "🧠\nGNN",
        "📈\nEvaluation",
        "🖥️\nDashboard",
    ]

    cols = st.columns(19)

    col_index = 0

    for i, step in enumerate(workflow):

        with cols[col_index]:

            st.markdown(
                f"""
<div style="
border:1px solid #d9d9d9;
border-radius:12px;
padding:18px 8px;
text-align:center;
font-weight:600;
font-size:18px;
height:95px;
display:flex;
align-items:center;
justify-content:center;
background:white;
">
{step.replace(chr(10), "<br>")}
</div>
""",
                unsafe_allow_html=True,
            )

        col_index += 1

        if i != len(workflow) - 1:

            with cols[col_index]:

                st.markdown(
                    """
<div style="
text-align:center;
font-size:34px;
padding-top:25px;
font-weight:bold;
">
➜
</div>
""",
                    unsafe_allow_html=True,
                )

            col_index += 1

    st.markdown("---")

    # ==========================================================
    # Project Statistics
    # ==========================================================

    st.subheader("📌 Project Statistics")

    stat1, stat2, stat3, stat4 = st.columns(4)

    with stat1:

        st.markdown(
            """
##### Dataset

# 21,613

### Houses
"""
        )

    with stat2:

        st.metric(
            "Features",
            "27",
        )

    with stat3:

        st.metric(
            "Models",
            "2",
        )

    with stat4:

        st.markdown(
            """
##### Framework

# PyTorch

### Geometric
"""
        )

    st.markdown("---")

    # ==========================================================
    # Repository Overview
    # ==========================================================

    st.subheader("📚 Repository Overview")

    repository_1, repository_2, repository_3 = st.columns(3)

    with repository_1:

        st.metric(
            "GitHub Issues Completed",
            "17",
        )

    with repository_2:

        st.metric(
            "Development Weeks",
            "4",
        )

    with repository_3:

        st.markdown(
            """
##### Primary Framework

## PyTorch

### Geometric
"""
        )

    st.markdown("---")

    st.caption(
        "Version 1.0 • Geospatial Real Estate Valuation using Graph Neural Networks • Developed by Aman Shukla"
    )