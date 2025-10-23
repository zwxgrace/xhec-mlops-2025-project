# Use this module to code a `pickle_object` function. This will be useful to pickle the model (and encoder if need be).
from pathlib import Path
import pickle
from typing import Any


def pickle_object(
    obj: Any, path: str | Path, *, protocol: int = pickle.HIGHEST_PROTOCOL
) -> Path:
    """
    Serialize (pickle) any Python object to `path`.

    - Creates parent directories if they don't exist.
    - Uses highest pickle protocol by default.
    - Overwrites existing file.

    Parameters
    ----------
    obj : Any
        The Python object to serialize.
    path : str | Path
        Destination file path, e.g. "src/web_service/local_objects/model.pkl".
    protocol : int
        Pickle protocol (default: pickle.HIGHEST_PROTOCOL).

    Returns
    -------
    Path
        The resolved path where the object was saved.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(obj, f, protocol=protocol)
    return p
