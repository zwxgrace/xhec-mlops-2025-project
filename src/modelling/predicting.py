from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from .preprocessing import preprocess_for_inference, Preprocessor


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
