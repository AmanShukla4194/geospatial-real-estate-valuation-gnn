"""
Main Streamlit Dashboard.

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import DASHBOARD_TITLE

from app.components.dataset_section import (
    render_dataset_section,
)

from app.components.metrics_section import (
    render_metrics_section,
)

from app.components.map_section import (
    render_map_section,
)

from app.components.prediction_section import (
    render_prediction_section,
)

from app.components.summary_section import (
    render_summary_section,
)


# ============================================================
# Streamlit Configuration
# ============================================================

st.set_page_config(

    page_title=DASHBOARD_TITLE,

    layout="wide",

    initial_sidebar_state="expanded",
)

# ============================================================
# Sidebar
# ============================================================

# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("🏠 Project Information")

st.sidebar.success(
    "Geospatial Real Estate Valuation Dashboard"
)

st.sidebar.markdown("---")

st.sidebar.subheader("Dataset")

st.sidebar.write(
    "King County Housing Dataset"
)

st.sidebar.subheader("Models")

st.sidebar.write("• XGBoost")

st.sidebar.write("• Graph Neural Network")

st.sidebar.subheader("Technology")

st.sidebar.write("• Python")

st.sidebar.write("• Streamlit")

st.sidebar.write("• GeoPandas")

st.sidebar.write("• PyTorch Geometric")

st.sidebar.write("• Folium")

st.sidebar.markdown("---")

st.sidebar.info(
    "Developed by Aman Shukla"
)

# ============================================================
# Header
# ============================================================

st.title("🏠 Geospatial Real Estate Valuation")

st.caption(
    "Interactive Machine Learning Dashboard using Graph Neural Networks"
)

st.markdown(
"""
This dashboard demonstrates the complete end-to-end workflow for
predicting residential property prices using traditional Machine Learning
and Graph Neural Networks while incorporating spatial relationships
between neighbouring properties.

Use the sections below to explore the dataset, compare model performance,
visualize housing locations, and analyze property characteristics.
"""
)

information_1, information_2, information_3 = st.columns(3)

information_1.info(
    "📍 King County Housing Dataset"
)

information_2.info(
    "🧠 XGBoost + Graph Neural Network"
)

information_3.info(
    "🗺 Interactive Geospatial Dashboard"
)

st.divider()

# ============================================================
# Dashboard Sections
# ============================================================

render_dataset_section()

st.divider()

render_metrics_section()

st.divider()

render_map_section()

st.divider()

render_prediction_section()

st.divider()

render_summary_section()