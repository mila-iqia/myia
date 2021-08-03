"""Supplementary inferrers for testing.

Help to test master operations.
Inferrers might be moved to standar inferrers later if correctly tested.
"""

import math

from myia import inferrers
from myia.abstract import data
from myia.infer.algo import Require
from myia.infer.inferrers import X
from myia.infer.infnode import (
    dispatch_inferences,
    inference_function,
    signature,
)
from myia.testing.common import Float, Int, Nil, Object, tuple_of
from myia.testing.master_placeholders import (
    dict_setitem,
    dict_values,
    tuple_setitem,
    zeros_like,
)


def dict_setitem_inferrer(node, args, unif, inferrers):
    """Inferrer for the dict_setitem function."""
    dct_node, key_node, value_node = args
    dct_type = yield Require(dct_node)
    key_type = yield Require(key_node)
    value_type = yield Require(value_node)
    assert isinstance(dct_type, data.AbstractDict)
    keys = dct_type.keys
    values = dct_type.values
    key_pos = keys.index(key_type)
    values[key_pos] = value_type
    return data.AbstractDict([el for item in zip(keys, values) for el in item])


def dict_values_inferrer(node, args, unif, inferrers):
    """Inferrer for the dict_values function."""
    (dct_node,) = args
    dct_type = yield Require(dct_node)
    assert isinstance(dct_type, data.AbstractDict)
    return data.AbstractStructure(dct_type.values, {"interface": tuple})


def tuple_setitem_inferrer(node, args, unif, inferrers):
    t_node, idx_node, v_node = args
    t_type = yield Require(t_node)
    idx_type = yield Require(idx_node)
    v_type = yield Require(v_node)
    # Check tuple type
    assert isinstance(t_type, data.AbstractStructure)
    assert t_type.tracks.interface is tuple
    # Check idx type
    assert isinstance(idx_type, data.AbstractAtom)
    assert idx_type.tracks.interface is int, f"Expected int index, got {idx_type.tracks.interface}"
    assert isinstance(idx_type.tracks.value, int), f"Expected int value, got {idx_type.tracks.value}"
    idx = idx_type.tracks.value
    output_elements = list(t_type.elements)
    output_elements[idx] = v_type
    return data.AbstractStructure(output_elements, {"interface": tuple})


def add_testing_inferrers():
    """Add supplementary inferrers for testing."""
    inferrers.update(
        {
            # master operation inferrers
            dict_setitem: inference_function(dict_setitem_inferrer),
            dict_values: inference_function(dict_values_inferrer),
            tuple_setitem: inference_function(tuple_setitem_inferrer),
            zeros_like: dispatch_inferences((int, 0), (float, 0.0)),
            math.log: dispatch_inferences(
                (bool, Float), (Int, Float), (Float, float)
            ),
            # type inferrers
            object: signature(ret=Object),
            type(None): signature(ret=Nil),
        }
    )
