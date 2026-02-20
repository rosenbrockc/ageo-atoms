import numpy as np
import icontract
from typing import Sequence, Union, Any

# Types for numpy
DTypeLike = Union[np.dtype, type, str, None]
ShapeLike = Union[int, Sequence[int]]
ArrayLike = Union[np.ndarray, Sequence[Any], float, int]

def _check_dot_dims(a: ArrayLike, b: ArrayLike) -> bool:
    """Check if dimensions of a and b are compatible for dot product."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.ndim == 0 or b_arr.ndim == 0:
        return True
    if a_arr.ndim == 1 and b_arr.ndim == 1:
        return a_arr.shape[0] == b_arr.shape[0]
    if a_arr.ndim == 2 and b_arr.ndim == 2:
        return a_arr.shape[1] == b_arr.shape[0]
    if a_arr.ndim == 2 and b_arr.ndim == 1:
        return a_arr.shape[1] == b_arr.shape[0]
    if a_arr.ndim == 1 and b_arr.ndim == 2:
        return a_arr.shape[0] == b_arr.shape[0]
    return False

@icontract.require(lambda object: object is not None, "Object must not be None")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "Result must be a numpy array")
def array(
    object: ArrayLike,
    dtype: DTypeLike = None,
    copy: bool = True,
    order: str = "K",
    subok: bool = False,
    ndmin: int = 0,
) -> np.ndarray:
    """Create an array.

    Args:
        object: An array, any object exposing the array interface, an
            object whose __array__ method returns an array, or any
            (nested) sequence.
        dtype: The desired data-type for the array. If None, then the
            type will be determined as the minimum type required to
            hold the objects in the sequence.
        copy: If true (default), then the object is copied. Otherwise,
            a copy will only be made if __array__ returns a copy, if
            obj is a nested sequence, or if a copy is needed to
            satisfy any of the other requirements.
        order: Specify the memory layout of the array.
        subok: If True, then sub-classes will be passed-through,
            otherwise the returned array will be forced to be a
            base-class array (default).
        ndmin: Specifies the minimum number of dimensions that the
            resulting array should have.

    Returns:
        An array object satisfying the specified requirements.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "N-Dimensional Tensor Constructor",
        "conceptual_transform": "Initializes a structured tensor representation from a raw sequence of elements or another compatible data structure. It defines the initial memory layout, data type, and minimal dimensionality of the resulting tensor.",
        "abstract_inputs": [
            {
                "name": "object",
                "description": "A raw sequence, nested collection, or existing tensor to be converted."
            },
            {
                "name": "dtype",
                "description": "A categorical identifier for the numerical representation."
            },
            {
                "name": "ndmin",
                "description": "An integer specifying the minimum rank of the resulting tensor."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A structured N-dimensional tensor satisfying the specified constraints."
            }
        ],
        "algorithmic_properties": [
            "data-conversion",
            "memory-allocation",
            "deterministic"
        ],
        "cross_disciplinary_applications": [
            "Converting raw sensor measurements into a vector for signal processing.",
            "Loading experimental results into a matrix for statistical analysis.",
            "Initializing a grid of spatial coordinates for a physical simulation."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return np.array(
        object,
        dtype=dtype,
        copy=copy,
        order=order,
        subok=subok,
        ndmin=ndmin,
    )

@icontract.require(lambda shape: isinstance(shape, (int, tuple, list)), "Shape must be an int or a sequence of ints")
@icontract.ensure(lambda result, shape: result.shape == (shape if isinstance(shape, tuple) else (shape,) if isinstance(shape, int) else tuple(shape)), "Result shape must match requested shape")
def zeros(shape: ShapeLike, dtype: DTypeLike = float, order: str = "C") -> np.ndarray:
    """Return a new array of given shape and type, filled with zeros.

    Args:
        shape: Shape of the new array, e.g., (2, 3) or 2.
        dtype: The desired data-type for the array, e.g., numpy.int8.
            Default is numpy.float64.
        order: Whether to store multi-dimensional data in row-major
            (C-style) or column-major (Fortran-style) order in memory.

    Returns:
        Array of zeros with the given shape, dtype, and order.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Null-Initialized Tensor Generator",
        "conceptual_transform": "Allocates and initializes a new N-dimensional tensor of a specified shape and type where every element is set to the additive identity (zero).",
        "abstract_inputs": [
            {
                "name": "shape",
                "description": "A tuple of integers defining the size of each dimension."
            },
            {
                "name": "dtype",
                "description": "The desired numerical representation."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tensor of the specified shape populated with null values."
            }
        ],
        "algorithmic_properties": [
            "tensor-initialization",
            "memory-allocation",
            "deterministic"
        ],
        "cross_disciplinary_applications": [
            "Initializing an empty accumulator for a numerical integration task.",
            "Allocating a buffer for pre-calculating state transitions in a model.",
            "Creating a baseline 'dark frame' for an image sensor calibration."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return np.zeros(shape, dtype=dtype, order=order)

@icontract.require(lambda a, b: _check_dot_dims(a, b), "Dimensions of a and b must be compatible for dot product")
@icontract.ensure(lambda result: result is not None, "Result must not be None")
def dot(a: ArrayLike, b: ArrayLike, out: np.ndarray | None = None) -> Any:
    """Dot product of two arrays.

    For 2-D arrays it is equivalent to matrix multiplication, and for
    1-D arrays to inner product of vectors.

    Args:
        a: First argument.
        b: Second argument.
        out: Output argument. This must have the exact kind that would
            be returned if it was not used.

    Returns:
        Returns the dot product of a and b. If a and b are both
        scalars or both 1-D arrays then a scalar is returned;
        otherwise an array is returned.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Multi-Dimensional Inner Product Resolver",
        "conceptual_transform": "Computes the generalized inner product of two tensors. For 1D tensors, it calculates the scalar sum of element-wise products; for 2D tensors, it performs standard linear operator composition.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "The first input tensor."
            },
            {
                "name": "b",
                "description": "The second input tensor, whose leading dimension must match the trailing dimension of 'a'."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "The resulting scalar or tensor representing the combined interaction of 'a' and 'b'."
            }
        ],
        "algorithmic_properties": [
            "linear-operator",
            "reduction",
            "algebraic-composition"
        ],
        "cross_disciplinary_applications": [
            "Applying a linear transformation to a state vector in control theory.",
            "Calculating the similarity between two high-dimensional feature vectors.",
            "Projecting a physical force vector onto a specific axis of motion."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return np.dot(a, b, out=out)

@icontract.require(lambda tup: len(tup) > 0, "Sequence of arrays must not be empty")
@icontract.ensure(lambda result, tup: result.shape[0] == sum(np.asarray(x).shape[0] if np.asarray(x).ndim > 1 else 1 for x in tup), "Result leading dimension must match sum of input leading dimensions")
def vstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike = None,
    casting: str = "same_kind",
) -> np.ndarray:
    """Stack arrays in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after 1-D
    arrays of shape (N,) have been reshaped to (1,N).

    Args:
        tup: Sequence of arrays. The arrays must have the same shape
            along all but the first axis. 1-D arrays must have the
            same length.
        dtype: If provided, the destination array will have this dtype.
        casting: Controls what kind of data casting may occur.

    Returns:
        The array formed by stacking the given arrays, which will be
        at least 2-D.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Primary-Dimension Tensor Concatenator",
        "conceptual_transform": "Joins a sequence of tensors along their first dimension, effectively stacking them 'vertically'. It requires all input tensors to have compatible secondary dimensions.",
        "abstract_inputs": [
            {
                "name": "tup",
                "description": "A sequence of compatible tensors."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A single tensor whose first dimension size is the sum of the input dimensions."
            }
        ],
        "algorithmic_properties": [
            "structural-aggregation",
            "data-join",
            "shape-manipulation"
        ],
        "cross_disciplinary_applications": [
            "Combining multiple independent sensor logs into a single unified dataset.",
            "Stacking individual image frames into a video volume.",
            "Aggregating multiple sub-models into a larger composite system."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return np.vstack(tup, dtype=dtype, casting=casting)

@icontract.require(lambda a: a is not None, "Array must not be None")
@icontract.ensure(lambda result, a: result.size == a.size, "Result size must match original array size")
def reshape(a: np.ndarray, newshape: ShapeLike, order: str = "C") -> np.ndarray:
    """Gives a new shape to an array without changing its data.

    Args:
        a: Array to be reshaped.
        newshape: The new shape should be compatible with the original
            shape. If an integer, then the result will be a 1-D array
            of that length. One shape dimension can be -1. In this
            case, the value is inferred from the length of the array
            and remaining dimensions.
        order: Read the elements of a using this index order, and
            place the elements into the reshaped array using this
            index order. 'C' means to read / write the elements using
            C-like index order.

    Returns:
        This will be a new view object if possible; otherwise, it will
        be a copy.
    

    <!-- conceptual_profile -->
    {
        "abstract_name": "Topological Dimension Reinterpreter",
        "conceptual_transform": "Changes the dimensional interpretation (shape) of a tensor without modifying the underlying elements or their relative order. It maps the same set of elements to a new coordinate system.",
        "abstract_inputs": [
            {
                "name": "a",
                "description": "The source tensor."
            },
            {
                "name": "newshape",
                "description": "The target dimensional structure (tuple of integers)."
            }
        ],
        "abstract_outputs": [
            {
                "name": "result",
                "description": "A tensor with the same total number of elements but a different rank or shape."
            }
        ],
        "algorithmic_properties": [
            "view-transformation",
            "shape-reinterpretation",
            "data-reformatting"
        ],
        "cross_disciplinary_applications": [
            "Flattening a 2D image into a 1D vector for machine learning input.",
            "Re-interpreting a long sensor stream as a series of fixed-width windowed segments.",
            "Converting a 3D volume into a 2D slice representation."
        ]
    }
    <!-- /conceptual_profile -->
    """
    return np.reshape(a, newshape, order=order)
