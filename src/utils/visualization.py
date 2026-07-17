"""
Visualization utility functions.

Reusable visualization functions for the Geospatial Real Estate
Valuation project.

Author:
    Aman Shukla
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import REPORTS_DIR

# =============================================================================
# Output Directories
# =============================================================================

FIGURE_DIR = REPORTS_DIR / "figures"

DISTRIBUTION_DIR = FIGURE_DIR / "distributions"
BOXPLOT_DIR = FIGURE_DIR / "boxplots"
CORRELATION_DIR = FIGURE_DIR / "correlation"

for directory in [
    FIGURE_DIR,
    DISTRIBUTION_DIR,
    BOXPLOT_DIR,
    CORRELATION_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Save Figure
# =============================================================================

def save_figure(output_path: Path):

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =============================================================================
# Histogram
# =============================================================================

def plot_histogram(
    dataframe: pd.DataFrame,
    column: str,
    bins: int = 30,
):

    plt.figure(figsize=(8, 5))

    plt.hist(
        dataframe[column],
        bins=bins
    )

    plt.title(f"{column} Distribution")

    plt.xlabel(column)

    plt.ylabel("Frequency")

    save_figure(
        DISTRIBUTION_DIR / f"{column}_distribution.png"
    )


# =============================================================================
# Boxplot
# =============================================================================

def plot_boxplot(
    dataframe: pd.DataFrame,
    column: str,
):

    plt.figure(figsize=(8, 2.5))

    plt.boxplot(
        dataframe[column],
        vert=False
    )

    plt.title(f"{column} Boxplot")

    save_figure(
        BOXPLOT_DIR / f"{column}_boxplot.png"
    )


# =============================================================================
# Correlation Heatmap
# =============================================================================

def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
):

    plt.figure(figsize=(12, 10))

    plt.imshow(
        correlation_matrix,
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=90,
    )

    plt.yticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
    )

    plt.title("Correlation Heatmap")

    save_figure(
        CORRELATION_DIR / "correlation_heatmap.png"
    )