import numpy as np
import icontract
from typing import Sequence, Union, Optional, Any

# Types
ShapeLike = Union[int, Sequence[int]]
SeedLike = Union[int, np.random.SeedSequence, np.random.BitGenerator, np.random.Generator, None]

@icontract.require(lambda size: size is None or isinstance(size, (int, tuple, list)), "Size must be None, an int, or a sequence of ints")
@icontract.ensure(lambda result, size: (result.shape == (size if isinstance(size, tuple) else (size,) if isinstance(size, int) else tuple(size))) if size is not None else isinstance(result, float), "Result shape must match requested size")
def rand(size: ShapeLike | None = None) -> Union[float, np.ndarray]:
    """Random values in a given shape.

    Create an array of the given shape and populate it with random
    samples from a uniform distribution over [0, 1).

    Args:
        size: The dimensions of the returned array, should be
            non-negative. If None, a single Python float is returned.

    Returns:
        Random values.
    """
    if size is None:
        return np.random.rand()
    if isinstance(size, int):
        return np.random.rand(size)
    return np.random.rand(*size)

@icontract.require(lambda low, high: low <= high, "low must be less than or equal to high")
@icontract.ensure(lambda result, size: (result.shape == (size if isinstance(size, tuple) else (size,) if isinstance(size, int) else tuple(size))) if size is not None else isinstance(result, (float, np.floating)), "Result shape must match requested size")
def uniform(
    low: float = 0.0,
    high: float = 1.0,
    size: ShapeLike | None = None,
) -> Union[float, np.floating, np.ndarray]:
    """Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval
    [low, high) (includes low, but excludes high).

    Args:
        low: Lower boundary of the output interval. All values
            generated will be greater than or equal to low.
        high: Upper boundary of the output interval. All values
            generated will be less than high.
        size: Output shape. If the given shape is, e.g., (m, n, k),
            then m * n * k samples are drawn. If size is None (default),
            a single value is returned if low and high are both
            scalars.

    Returns:
        Drawn samples from the parameterized uniform distribution.
    """
    return np.random.uniform(low, high, size)

@icontract.require(lambda seed: seed is None or isinstance(seed, (int, np.random.SeedSequence, np.random.BitGenerator, np.random.Generator)) or (isinstance(seed, Sequence) and not isinstance(seed, str)), "Invalid seed type")
@icontract.ensure(lambda result: isinstance(result, np.random.Generator), "Result must be a numpy Generator")
def default_rng(seed: SeedLike = None) -> np.random.Generator:
    """Construct a new Generator with the default BitGenerator (PCG64).

    Args:
        seed: Reseed the BitGenerator. If None, then fresh,
            unpredictable entropy will be pulled from the OS.

    Returns:
        The initialized generator object.
    """
    return np.random.default_rng(seed)
