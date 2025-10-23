from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from .preprocessing import preprocess_for_inference, Preprocessor

from prefect import task


@task(name="predict-batch", tags={"prediction", "inference", "ml"})
def predict_batch(
    data_path: str,
    model_path: str,
    preproc_path: str,
) -> np.ndarray:
    """
    return (np.ndarray)
    """
    df = pd.read_csv(data_path)

    model = joblib.load(model_path)
    preproc: Preprocessor = joblib.load(preproc_path)

    X_df = preprocess_for_inference(df, preproc)
    y_pred = model.predict(X_df.values)
    return y_pred


@task(name="Evaluate model")
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error for two arrays"""
    return mean_squared_error(y_true, y_pred, squared=False)
