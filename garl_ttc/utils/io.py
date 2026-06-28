import pickle
import sys
from typing import List

import cv2
import numpy as np


def _register_numpy_pickle_aliases() -> None:
    """Allow pkl files written by NumPy 2.x to load under NumPy 1.x."""
    try:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core._multiarray_umath as numpy_multiarray_umath
    except Exception:
        return
    sys.modules.setdefault('numpy._core', numpy_core)
    sys.modules.setdefault('numpy._core.multiarray', numpy_multiarray)
    sys.modules.setdefault('numpy._core._multiarray_umath', numpy_multiarray_umath)


def read_image(path: str) -> np.ndarray:
    return cv2.imread(path)


def read_txt(path: str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    return [item.rstrip() for item in lines]


def read_pkl(path: str) -> list:
    _register_numpy_pickle_aliases()
    with open(path, 'rb') as f:
        return pickle.load(f)
