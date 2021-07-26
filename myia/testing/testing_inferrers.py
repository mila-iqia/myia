"""Supplementary inferrers for testing.

Help to test master operations.
Inferrers might be moved to standar inferrers later if correctly tested.
"""

from myia import inferrers
from myia.abstract import data
from myia.infer.algo import Require
from myia.infer.inferrers import X
from myia.infer.infnode import InferenceEngine, inference_function, signature
from myia.testing import numpy_subset, numpy_subset as np
from myia.testing.common import (
    Float,
    Integer,
    Nil,
    Number,
    Object,
    array_of,
    tuple_of,
)
from myia.testing.master_placeholders import (
    array_cast,
    array_to_scalar,
    distribute,
    dot,
    reshape,
    shape,
    transpose,
    tuple_setitem,
)


def np_full_inferrer(node, args, unif):
    """Inferrer for the numpy.full function."""
    shape_node, value_node, dtype_node = args
    # shape_type = yield Require(shape_node)
    value_type = yield Require(value_node)
    dtype = yield Require(dtype_node)
    if dtype is Nil:
        dtype = value_type
    elif isinstance(dtype, data.AbstractAtom) and dtype.tracks.interface is str:
        type_name = dtype.tracks.value
        try:
            dtype = data.AbstractAtom({"interface": np.dtype(type_name).type})
        except TypeError:
            raise TypeError(f"Cannot parse numpy dtype {type_name}")
    elif InferenceEngine.is_abstract_type(dtype):
        dtype = dtype.elements[0]
    else:
        raise TypeError(f"Expected an abstract type, got {dtype}")
    return data.AbstractStructure([dtype], {"interface": np.ndarray})


def add_testing_inferrers():
    """Add supplementary inferrers for testing."""
    inferrers.update(
        {
            # master operation inferrers
            array_cast: signature(array_of(), X, ret=array_of(X)),
            array_to_scalar: signature(array_of(X), ret=X),
            distribute: signature(array_of(), tuple_of(), ret=array_of()),
            dot: signature(array_of(), array_of(), ret=array_of()),
            # todo
            reshape: signature(array_of(), tuple_of(), ret=array_of()),
            shape: signature(array_of(), ret=tuple_of()),
            transpose: signature(array_of(), tuple_of(), ret=array_of()),
            tuple_setitem: signature(tuple_of(), Integer, X, ret=tuple_of()),
            # numpy subset inferrers
            numpy_subset.array: signature(Number, ret=array_of(Number, ())),
            numpy_subset.full: inference_function(np_full_inferrer),
            numpy_subset.log: signature(Number, ret=Float),
            numpy_subset.prod: signature(
                array_of(Number), ret=array_of(Number, ())
            ),
            # type inferrers
            object: signature(ret=Object),
            type(None): signature(ret=Nil),
        }
    )
