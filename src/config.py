"""
Central configuration module for the Geospatial Real Estate Valuation project.

This module contains all project-wide constants, file paths, configuration
parameters, and reproducibility settings.

Author:
    Aman Shukla

Project:
    Geospatial Real Estate Valuation using Graph Neural Networks
"""

from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

APP_DIR = PROJECT_ROOT / "app"

DOCS_DIR = PROJECT_ROOT / "docs"

# =============================================================================
# Dataset
# =============================================================================

DATASET_FILENAME = "kc_house_data.csv"

DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME

TARGET_COLUMN = "price"

# =============================================================================
# Geographic Features
# =============================================================================

LATITUDE_COLUMN = "lat"

LONGITUDE_COLUMN = "long"

# =============================================================================
# Randomness
# =============================================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

# =============================================================================
# Required Dataset Columns
# =============================================================================

REQUIRED_COLUMNS = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
]

# =============================================================================
# Data Cleaning Configuration
# =============================================================================

OUTLIER_COLUMNS = [
    "price",
    "sqft_living",
    "sqft_lot",
    "sqft_above",
    "sqft_basement",
    "sqft_living15",
    "sqft_lot15",
]

NUMERIC_VALIDATION_COLUMNS = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "grade",
    "lat",
    "long",
]

# =============================================================================
# Baseline Machine Learning Configuration
# =============================================================================

BASELINE_FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "house_age",
    "is_renovated",
    "years_since_renovation",
    "distance_to_city_center_km",
]

# =============================================================================
# Graph Configuration
# =============================================================================

K_NEAREST_NEIGHBORS = 10

GRAPH_RANDOM_STATE = 42

# =============================================================================
# Graph Neural Network Configuration
# =============================================================================

NODE_FEATURE_COLUMNS = BASELINE_FEATURE_COLUMNS

GRAPH_TARGET_COLUMN = TARGET_COLUMN

# =============================================================================
# GNN Architecture
# =============================================================================

GNN_HIDDEN_CHANNELS = 64

GNN_OUTPUT_CHANNELS = 1

# =============================================================================
# GNN Training Configuration
# =============================================================================

GNN_LEARNING_RATE = 0.001

GNN_EPOCHS = 100

GNN_WEIGHT_DECAY = 1e-5

GNN_RANDOM_STATE = 42

# =============================================================================
# Model Output Paths
# =============================================================================

GNN_MODEL_PATH = MODELS_DIR / "property_gnn_model.pt"

TRAINING_HISTORY_PATH = MODELS_DIR / "training_history.csv"

# =============================================================================
# Evaluation Reports
# =============================================================================

MODEL_COMPARISON_REPORT = (
    REPORTS_DIR
    / "model_comparison_summary.md"
)

MODEL_COMPARISON_FIGURE = (
    REPORTS_DIR
    / "figures"
    / "model_comparison.png"
)