"""
Baseline Model Evaluation.

Evaluates the trained XGBoost baseline model.

Author:
    Aman Shukla
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from src.config import (
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    BASELINE_FEATURE_COLUMNS,
)

MODEL_FILE = Path(
    "models/baseline_xgboost.joblib"
)

ENGINEERED_DATA_FILE = (
    PROCESSED_DATA_DIR
    / "engineered_housing_data.csv"
)


def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    return joblib.load(
        MODEL_FILE
    )


def load_dataset():

    dataframe = pd.read_csv(
        ENGINEERED_DATA_FILE
    )

    return dataframe


def evaluate_model():

    print("=" * 70)

    print(
        "Evaluating Baseline XGBoost Model"
    )

    print("=" * 70)

    dataframe = load_dataset()

    x = dataframe[
        BASELINE_FEATURE_COLUMNS
    ]

    y = dataframe[
        TARGET_COLUMN
    ]

    from sklearn.model_selection import train_test_split

    (
        _,
        x_test,
        _,
        y_test,
    ) = train_test_split(

        x,

        y,

        test_size=0.20,

        random_state=42,
    )

    model = load_model()

    predictions = model.predict(
        x_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    mape = (
        mean_absolute_percentage_error(
            y_test,
            predictions,
        )
        * 100
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print(f"MAE  : {mae:,.2f}")

    print(f"RMSE : {rmse:,.2f}")

    print(f"MAPE : {mape:.2f}%")

    print(f"R²   : {r2:.4f}")

    print("=" * 70)

    return {

        "MAE": mae,

        "RMSE": rmse,

        "MAPE": mape,

        "R2": r2,

    }


if __name__ == "__main__":

    evaluate_model()