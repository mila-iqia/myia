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


def dict_setitem_inferrer(node, args, unif):
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


def dict_values_inferrer(node, args, unif):
    """Inferrer for the dict_values function."""
    (dct_node,) = args
    dct_type = yield Require(dct_node)
    assert isinstance(dct_type, data.AbstractDict)
    return data.AbstractStructure(dct_type.values, {"interface": tuple})


def add_testing_inferrers():
    """Add supplementary inferrers for testing."""
    inferrers.update(
        {
            # master operation inferrers
            dict_setitem: inference_function(dict_setitem_inferrer),
            dict_values: inference_function(dict_values_inferrer),
            tuple_setitem: signature(tuple_of(), Int, X, ret=tuple_of()),
            zeros_like: dispatch_inferences((int, 0), (float, 0.0)),
            math.log: dispatch_inferences(
                (bool, Float), (Int, Float), (Float, float)
            ),
            # type inferrers
            object: signature(ret=Object),
            type(None): signature(ret=Nil),
        }
    )
