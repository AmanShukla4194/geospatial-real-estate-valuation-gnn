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

st.sidebar.title(
    "Navigation"
)

st.sidebar.info(
    """
    Geospatial Real Estate Valuation

    Week 4 Dashboard
    """
)

# ============================================================
# Header
# ============================================================

st.title(
    DASHBOARD_TITLE
)

st.markdown(
    """
    Interactive dashboard demonstrating the complete
    machine learning pipeline developed during the
    Geospatial Real Estate Valuation internship project.
    """
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