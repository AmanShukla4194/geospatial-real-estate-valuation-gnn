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

from src.project_info import (
    PROJECT_NAME,
    PROJECT_SUBTITLE,
    PROJECT_DESCRIPTION,
    DATASET_NAME,
    BASELINE_MODEL,
    GRAPH_MODEL,
    AUTHOR,
)

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

st.sidebar.title("🏠 Project Information")

st.sidebar.success(
    f"{PROJECT_NAME} Dashboard"
)

st.sidebar.markdown("---")

st.sidebar.subheader("Dataset")

st.sidebar.write(
    DATASET_NAME
)

st.sidebar.subheader("Models")

st.sidebar.write(f"• {BASELINE_MODEL}")

st.sidebar.write(f"• {GRAPH_MODEL}")

st.sidebar.subheader("Technology")

st.sidebar.write("• Python")

st.sidebar.write("• Streamlit")

st.sidebar.write("• GeoPandas")

st.sidebar.write("• PyTorch Geometric")

st.sidebar.write("• Folium")

st.sidebar.write("• Plotly")

st.sidebar.markdown("---")

st.sidebar.info(
    f"Developed by {AUTHOR}"
)

# ============================================================
# Header
# ============================================================

st.title(
    f"🏠 {PROJECT_NAME}"
)

st.caption(
    PROJECT_SUBTITLE
)

st.markdown(
    PROJECT_DESCRIPTION
)

information_1, information_2, information_3 = st.columns(3)

information_1.info(
    f"📍 {DATASET_NAME}"
)

information_2.info(
    f"🧠 {BASELINE_MODEL} + GNN"
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