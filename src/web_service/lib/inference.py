# src/web_service/lib/inference.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from ..utils import load_object
from .models import PredictionInput

from ...modelling.preprocessing import Preprocessor, preprocess_for_inference

# define paths to the model and preprocessor artifacts
BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_OBJECTS_DIR = BASE_DIR / "local_objects"
MODEL_PATH = LOCAL_OBJECTS_DIR / "model.pkl"
PREPROCESSOR_PATH = LOCAL_OBJECTS_DIR / "preprocessor.pkl"

# load the model and preprocessor at module load time
try:
    print("Loading model...")
    model = load_object(MODEL_PATH)
    print("Model loaded successfully.")

    print("Loading preprocessor...")
    preprocessor: Preprocessor = load_object(PREPROCESSOR_PATH)
    print("Preprocessor loaded successfully.")

except Exception as e:
    print(f"Fatal error loading artifacts: {e}")
    model = None
    preprocessor = None

COLUMN_NAME_MAPPING = {
    "Whole_weight": "Whole weight",
    "Shucked_weight": "Shucked weight",
    "Viscera_weight": "Viscera weight",
    "Shell_weight": "Shell weight",
}


def make_prediction(payload: PredictionInput) -> dict:
    """
    Performs the full inference pipeline:
    1. Converts Pydantic input to DataFrame
    2. Renames columns
    3. Preprocesses data
    4. Makes prediction
    """
    if not model or not preprocessor:
        return {"error": "Model or preprocessor not loaded."}

    input_df = pd.DataFrame([payload.model_dump()])

    try:
        input_df = input_df.rename(columns=COLUMN_NAME_MAPPING)
    except Exception as e:
        return {"error": f"DataFrame renaming failed: {e}"}

    try:
        X_processed_df = preprocess_for_inference(input_df, preprocessor)

    except KeyError as e:
        return {
            "error": f"Preprocessing failed: Missing columns {e}. DF columns are: {list(input_df.columns)}"
        }
    except Exception as e:
        return {"error": f"Preprocessing failed: {e}"}

    # try to make prediction
    try:
        prediction = model.predict(X_processed_df.values)
        predicted_value = float(prediction[0])
        return {"predicted_rings": predicted_value}

    except Exception as e:
        return {"error": f"Prediction failed: {e}"}
