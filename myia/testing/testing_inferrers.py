"""Supplementary inferrers for testing.

Help to test master operations.
Inferrers might be moved to standar inferrers later if correctly tested.
"""

import math
from typing import Tuple, Sequence, Any, Dict
from myia import inferrers
from myia.abstract.to_abstract import type_to_abstract
from myia.abstract.data import AbstractValue
from myia.abstract import utils as autils
from myia.infer.inferrers import X, inference_function
from myia.infer.infnode import signature, Require
from myia.testing.common import Float, Int, Nil, Number, Object, tuple_of, Bool, Un
from myia.testing.master_placeholders import tuple_setitem


class InferenceDefinition:
    __slots__ = "arg_types", "ret_type"

    def __init__(self, *arg_types, ret_type):
        self.arg_types = tuple(type_to_abstract(arg_type) if not isinstance(arg_type, AbstractValue) else arg_type for arg_type in arg_types)
        self.ret_type = type_to_abstract(ret_type) if not isinstance(ret_type, AbstractValue) else ret_type


def dispatch_inferences(*signatures: Sequence):
    """Create an inference function from many type signatures.

    Arguments:
        signatures: a sequence of type signatures.
            Each signature is a sequence of types or abstract values.
            First sequence values are the argument types.
            Last sequence value is the return type.
            Each sequence must contain at least one element (the return type).
    """
    def_map = {}  # type: Dict[int, Dict[Tuple, InferenceDefinition]]
    for sig in signatures:
        if not isinstance(sig, InferenceDefinition):
            *arg_types, ret_type = sig
            sig = InferenceDefinition(*arg_types, ret_type=ret_type)
        def_map.setdefault(len(sig.arg_types), {})[sig.arg_types] = sig

    def inference(node, args, unif):
        inp_types = []
        for inp in args:
            inp_types.append((yield Require(inp)))
        inp_types = tuple(inp_types)
        inf_def = def_map.get(len(inp_types), {}).get(inp_types, None)
        if not inf_def:
            raise RuntimeError(f"No inference for node: {node}, signature: {inp_types}")
        for inp_type, expected_type in zip(inp_types, inf_def.arg_types):
            autils.unify(inp_type, expected_type, U=unif)
        return autils.reify(inf_def.ret_type, unif=unif.canon)

    return inference_function(inference)


def add_testing_inferrers():
    """Add supplementary inferrers for testing."""
    inferrers.update(
        {
            # master operation inferrers
            tuple_setitem: signature(tuple_of(), Int, X, ret=tuple_of()),
            math.log: dispatch_inferences(
                (bool, Float), (Int, Float), (Float, float)
            ),
            # type inferrers
            object: signature(ret=Object),
            type(None): signature(ret=Nil),
        }
    )
