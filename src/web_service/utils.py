from __future__ import annotations
import joblib
from pathlib import Path
from typing import Any


def load_object(path: str | Path) -> Any:
    """
    Load a pickled object from the given path.
    """
    try:
        return joblib.load(path)
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None
    except Exception as e:
        print(f"Error loading object from {path}: {e}")
        return None
