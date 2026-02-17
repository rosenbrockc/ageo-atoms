import numpy as np
import icontract
from typing import Any, Sequence, Union

@icontract.require(lambda object: object is not None, "Object must not be None")
def array(object: Any, dtype: Any = None, **kwargs: Any) -> np.ndarray:
    """
    Create an array.

    Args:
        object: An array, any object exposing the array interface, an object whose __array__ method returns an array, or any (nested) sequence.
        dtype: The desired data-type for the array.

    Returns:
        An array object satisfying the specified requirements.
    """
    return np.array(object, dtype=dtype, **kwargs)

@icontract.require(lambda shape: isinstance(shape, (int, tuple, list)), "Shape must be an int or a sequence of ints")
def zeros(shape: Union[int, Sequence[int]], dtype: Any = float, order: str = "C") -> np.ndarray:
    """
    Return a new array of given shape and type, filled with zeros.

    Args:
        shape: Shape of the new array.
        dtype: The desired data-type for the array.
        order: Whether to store multi-dimensional data in row-major (C-style) or column-major (Fortran-style) order in memory.

    Returns:
        Array of zeros with the given shape, dtype, and order.
    """
    return np.zeros(shape, dtype=dtype, order=order)

@icontract.require(lambda a, b: True, "Placeholder for dot product constraints")
def dot(a: Any, b: Any, out: Any = None) -> Any:
    """
    Dot product of two arrays.

    Args:
        a: First argument.
        b: Second argument.
        out: Output argument.

    Returns:
        Returns the dot product of a and b.
    """
    return np.dot(a, b, out=out)

from . import linalg

from . import random
