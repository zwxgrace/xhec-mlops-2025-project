from fastapi import FastAPI, HTTPException
from .lib.models import PredictionInput, PredictionOutput
from .lib.inference import make_prediction

app = FastAPI(
    title="Abalone Age Prediction API",
    description="API to predict the age (number of rings) of abalone.",
)


@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=PredictionOutput, status_code=201)
def predict(payload: PredictionInput) -> dict:
    """
    Prediction endpoint.
    Takes raw features, returns predicted rings.
    """
    result = make_prediction(payload)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result
