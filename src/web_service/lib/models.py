from pydantic import BaseModel


class PredictionInput(BaseModel):
    """
    Pydantic model for input features (Python-friendly names).
    """

    Length: float
    Diameter: float
    Height: float
    Whole_weight: float
    Shucked_weight: float
    Viscera_weight: float
    Shell_weight: float
    Sex: str

    class Config:
        from_attributes = True


class PredictionOutput(BaseModel):
    """
    Pydantic model for the prediction output
    """

    predicted_rings: float
