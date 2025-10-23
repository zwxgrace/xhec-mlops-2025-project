from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .preprocessing import preprocess_for_training


def train_and_save(
    data_path: str,
    model_path: str,
    preproc_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Reading → preprocessing → delimitation → training → evaluation → pkl
    """
    df = pd.read_csv(data_path)

    X_df, y, preproc = preprocess_for_training(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state
    )
    model = LinearRegression()
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    rmse = mean_squared_error(y_te, y_pred, squared=False)
    r2 = r2_score(y_te, y_pred)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(preproc_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(preproc, preproc_path)

    return {
        "rmse": float(rmse),
        "r2": float(r2),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    }
