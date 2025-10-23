from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_FEATURES: List[str] = [
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
]
CAT_FEATURES: List[str] = ["Sex"]
TARGET: str = "Rings"


@dataclass
class Preprocessor:
    """Use for transform"""

    column_transformer: ColumnTransformer

    def transform_to_df(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = self.column_transformer.transform(X)
        num_names = NUM_FEATURES
        cat_encoder: OneHotEncoder = self.column_transformer.named_transformers_[
            "cat"
        ].named_steps["oh"]
        cat_names = list(cat_encoder.get_feature_names_out(CAT_FEATURES))
        cols = num_names + cat_names
        return pd.DataFrame(Xt, columns=cols, index=X.index)


def _build_column_transformer() -> ColumnTransformer:
    num_tf = Pipeline(steps=[("scaler", StandardScaler())])
    cat_tf = Pipeline(steps=[("oh", OneHotEncoder(handle_unknown="ignore"))])

    ct = ColumnTransformer(
        transformers=[
            ("num", num_tf, NUM_FEATURES),
            ("cat", cat_tf, CAT_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return ct


def preprocess_for_training(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, Preprocessor]:
    """
    return:
    - X_df
    - y
    - preproc
    """
    X_raw = df[NUM_FEATURES + CAT_FEATURES].copy()
    y = df[TARGET].copy()

    ct = _build_column_transformer()
    ct.fit(X_raw)

    preproc = Preprocessor(column_transformer=ct)
    X_df = preproc.transform_to_df(X_raw)
    return X_df, y, preproc


def preprocess_for_inference(df: pd.DataFrame, preproc: Preprocessor) -> pd.DataFrame:
    """
    original df (may not contain TARGET) and returns a DataFrame of features with columns consistent with those at training
    """
    X_raw = df[NUM_FEATURES + CAT_FEATURES].copy()
    X_df = preproc.transform_to_df(X_raw)
    return X_df
