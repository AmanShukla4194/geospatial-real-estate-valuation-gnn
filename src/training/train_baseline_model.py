"""
Baseline XGBoost Training Pipeline.

This module trains the baseline XGBoost regression model for the
Geospatial Real Estate Valuation project.

Author:
    Aman Shukla
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from src.config import (
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    BASELINE_FEATURE_COLUMNS,
)


ENGINEERED_DATA_FILE = (
    PROCESSED_DATA_DIR
    / "engineered_housing_data.csv"
)

MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_FILE = (
    MODEL_DIR
    / "baseline_xgboost.joblib"
)


def load_dataset() -> pd.DataFrame:
    """
    Load engineered dataset.
    """

    if not ENGINEERED_DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{ENGINEERED_DATA_FILE}"
        )

    dataframe = pd.read_csv(
        ENGINEERED_DATA_FILE
    )

    print(
        f"Dataset loaded: "
        f"{len(dataframe):,} rows."
    )

    return dataframe


def prepare_data(
    dataframe: pd.DataFrame,
):
    """
    Prepare train/test data.
    """

    x = dataframe[
        BASELINE_FEATURE_COLUMNS
    ]

    y = dataframe[
        TARGET_COLUMN
    ]

    return train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
    )


def build_model():

    return XGBRegressor(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=6,

        subsample=0.80,

        colsample_bytree=0.80,

        objective="reg:squarederror",

        random_state=42,

        n_jobs=-1,
    )


def train_model():

    print("=" * 70)

    print(
        "Training XGBoost Baseline Model"
    )

    print("=" * 70)

    dataframe = load_dataset()

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = prepare_data(
        dataframe
    )

    print(
        f"Training Samples : {len(x_train):,}"
    )

    print(
        f"Testing Samples  : {len(x_test):,}"
    )

    model = build_model()

    print(
        "\nTraining..."
    )

    model.fit(
        x_train,
        y_train,
    )

    print(
        "Training completed."
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print(
        f"\nModel saved:\n{MODEL_FILE}"
    )

    predictions = model.predict(
        x_test
    )

    print("=" * 70)

    return (
        model,
        predictions,
        y_test,
        x_test,
    )


if __name__ == "__main__":

    train_model()