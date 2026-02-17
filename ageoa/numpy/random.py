import numpy as np
import icontract
from typing import Any, Optional, Sequence, Union

@icontract.require(lambda size: size is None or isinstance(size, (int, tuple, list)), "Size must be None, an int, or a sequence of ints")
def rand(*size: int) -> np.ndarray:
    """
    Random values in a given shape.

    Args:
        size: The dimensions of the returned array, must be non-negative.

    Returns:
        Random values.
    """
    return np.random.rand(*size)

@icontract.require(lambda low, high: low <= high, "low must be less than or equal to high")
def uniform(low: float = 0.0, high: float = 1.0, size: Optional[Union[int, Sequence[int]]] = None) -> Union[float, np.ndarray]:
    """
    Draw samples from a uniform distribution.

    Args:
        low: Lower boundary of the output interval.
        high: Upper boundary of the output interval.
        size: Output shape.

    Returns:
        Drawn samples from the parameterized uniform distribution.
    """
    return np.random.uniform(low, high, size)

@icontract.require(lambda seed: seed is None or isinstance(seed, (int, Sequence, np.random.SeedSequence, np.random.BitGenerator, np.random.Generator)), "Invalid seed type")
def default_rng(seed: Any = None) -> np.random.Generator:
    """
    Construct a new Generator with the default BitGenerator (PCG64).

    Args:
        seed: Reseed the BitGenerator.

    Returns:
        The initialized generator object.
    """
    return np.random.default_rng(seed)
