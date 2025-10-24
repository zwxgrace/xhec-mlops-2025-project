from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
from prefect import flow, get_run_logger
from prefect.artifacts import create_table_artifact

from .predicting import predict_batch
from .training import train_and_save


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "web_service" / "local_objects"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"


@flow(name="Train model")
def train_model_workflow(
    data_path: str,
    artifacts_filepath: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train a LinearRegression model from a single dataset, split internally."""
    logger = get_run_logger()
    logger.info("Starting training workflow")
    logger.info(f"Dataset: {data_path}")

    if artifacts_filepath is None:
        artifacts_filepath = (
            Path(__file__).resolve().parents[1] / "web_service" / "local_objects"
        )

    model_path = Path(artifacts_filepath) / "model.pkl"
    preproc_path = Path(artifacts_filepath) / "preprocessor.pkl"

    logger.info("Training model with internal train/test split...")
    result = train_and_save(
        data_path=data_path,
        model_path=str(model_path),
        preproc_path=str(preproc_path),
        test_size=test_size,
        random_state=random_state,
    )
    logger.info(f"RMSE: {result['rmse']:.4f} | R²: {result['r2']:.4f}")
    create_table_artifact(
        key="training-metrics",
        table=[
            {"metric": "rmse", "value": result["rmse"]},
            {"metric": "r2", "value": result["r2"]},
        ],
        description="Model performance metrics",
    )
    return {
        "rmse": result["rmse"],
        "r2": result["r2"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "model_path": str(model_path),
        "preprocessor_path": str(preproc_path),
    }


@flow(name="Batch predict", retries=1, retry_delay_seconds=30)
def predict_flow(
    input_filepath: str,
    model_path: str,
    preproc_path: str,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Make predictions on a new dataset using saved model & preprocessor.
    """
    y_pred = predict_batch(input_filepath, model_path, preproc_path)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"prediction": y_pred}).to_csv(out, index=False)
        print(f"Predictions saved to: {out}")

    print("Prediction flow completed.")
    return y_pred


if __name__ == "__main__":
    # Example with relative paths, assuming script is run from project root
    train_model_workflow(
        data_path="abalone.csv",
        artifacts_filepath="src/web_service/local_objects",
    )
