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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Standard Uniform Stochastic Tensor Generator",
        "conceptual_transform": "Generates an N-dimensional tensor populated with independent samples from a standard uniform probability distribution [0, 1). It provides a source of raw stochastic entropy for simulation and modeling.",
        "abstract_inputs": [
            {
                "name": "size",
                "description": "An optional tuple of integers defining the output dimensional structure."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tensor of the specified shape (or a scalar) containing stochastic values."
            }
        ],
        "algorithmic_properties": [
            "stochastic-generative",
            "uniform-distribution",
            "entropy-source"
        ],
        "cross_disciplinary_applications": [
            "Generating initial random weights for a neural network.",
            "Simulating random noise in a communication channel model.",
            "Monte Carlo sampling for approximating high-dimensional integrals."
        ]
    }
    <!-- /conceptual_profile -->
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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Parameterized Uniform Stochastic Tensor Generator",
        "conceptual_transform": "Generates an N-dimensional tensor populated with independent samples from a uniform probability distribution within a specified range [low, high). It scales and shifts standard stochastic entropy to a target interval.",
        "abstract_inputs": [
            {
                "name": "low",
                "description": "A scalar representing the lower bound of the distribution."
            },
            {
                "name": "high",
                "description": "A scalar representing the upper bound of the distribution."
            },
            {
                "name": "size",
                "description": "An optional tuple of integers defining the output dimensional structure."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tensor containing stochastic values within the requested range."
            }
        ],
        "algorithmic_properties": [
            "stochastic-generative",
            "uniform-distribution",
            "scaled-entropy"
        ],
        "cross_disciplinary_applications": [
            "Modeling uncertain sensor parameters within a known tolerance range.",
            "Randomizing the initial positions of agents in a spatial simulation.",
            "Generating test cases for an algorithm with bounded numerical inputs."
        ]
    }
    <!-- /conceptual_profile -->
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
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "High-Quality Stochastic Entropy Engine Constructor",
        "conceptual_transform": "Initializes a state-of-the-art pseudo-random number generator (PRNG) using a specified seed or OS-provided entropy. It provides a deterministic or non-deterministic root for all subsequent stochastic transformations.",
        "abstract_inputs": [
            {
                "name": "seed",
                "description": "An optional integer or entropy sequence used to initialize the internal PRNG state."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A generator object capable of producing high-quality stochastic sequences."
            }
        ],
        "algorithmic_properties": [
            "prng-initialization",
            "stateful-entropy-source",
            "deterministic-if-seeded"
        ],
        "cross_disciplinary_applications": [
            "Ensuring reproducible results in a stochastic scientific simulation.",
            "Initializing a secure sequence generator.",
            "Providing a central entropy root for a multi-threaded parallel simulation."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return np.random.default_rng(seed)
