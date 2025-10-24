# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression

from .preprocessing import preprocess_for_training
from .utils import pickle_object

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "web_service" / "local_objects"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"


def main(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model (pickle)."""
    # Read data
    df = pd.read_csv(trainset_path)
    # Preprocess data
    X_df, y, preproc = preprocess_for_training(df)
    # (Optional) Pickle encoder if need be
    pickle_object(preproc, PREPROCESSOR_PATH)
    # Train model
    model = LinearRegression()
    model.fit(X_df.values, y)
    # Pickle model --> The model should be saved in pkl format the `src/web_service/local_objects` folder
    pickle_object(model, MODEL_PATH)

    print(f"Saved preprocessor to: {PREPROCESSOR_PATH}")
    print(f"Saved model to:        {MODEL_PATH}")
    print(f"Train samples: {len(X_df)} | n_features: {X_df.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(Path(args.trainset_path))
