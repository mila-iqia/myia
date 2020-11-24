"""Definitions of computations.

A computation is a group of backend primitives that perform related operations.
To support a computation, a backend must implement all related primitives.
"""
from myia.operations import primitives as P

_COMPUTATIONS = {
    # Single computations
    "argmax": [P.argmax],
    "array_cast": [P.array_cast],
    "array_getitem": [P.array_getitem],
    "array_map": [P.array_map],
    "array_max": [P.array_max],
    "array_reduce": [P.array_reduce],
    "array_setitem": [P.array_setitem],
    "array_to_scalar": [P.array_to_scalar],
    "concat": [P.concat],
    "conv2d": [P.conv2d],
    "conv2d_weight_grad": [P.conv2d_weight_grad],
    "conv_transpose2d": [P.conv_transpose2d],
    "gather": [P.gather],
    "make_handle": [P.make_handle],
    "max_pool2d": [P.max_pool2d],
    "max_pool2d_grad": [P.max_pool2d_grad],
    "scatter": [P.scatter],
    "scatter_add": [P.scatter_add],
    "split": [P.split],
    "take": [P.take],
    "take_grad_inp": [P.take_grad_inp],
    "unsafe_static_cast": [P.unsafe_static_cast],
    # Computation groups
    "exception_operations": [P.raise_, P.make_exception],
    "tag_operations": [P.casttag, P.tagged, P.hastag],
    "env_operations": [P.env_getitem, P.env_setitem],
    "tuple_operations": [P.make_tuple, P.tuple_getitem, P.tuple_setitem],
    "universe_operations": [P.universe_getitem, P.universe_setitem],
    "rng_operations": [P.random_initialize, P.random_uint32],
    "logical_operations": [
        P.bool_and,
        P.bool_eq,
        P.bool_not,
        P.bool_or,
        P.switch,
    ],
    "array_operations": [P.distribute, P.dot, P.reshape, P.shape, P.transpose],
    "scalar_operations": [
        P.scalar_abs,
        P.scalar_add,
        P.scalar_bit_and,
        P.scalar_bit_lshift,
        P.scalar_bit_not,
        P.scalar_bit_or,
        P.scalar_bit_rshift,
        P.scalar_bit_xor,
        P.scalar_cast,
        P.scalar_cos,
        P.scalar_div,
        P.scalar_eq,
        P.scalar_exp,
        P.scalar_floor,
        P.scalar_ge,
        P.scalar_gt,
        P.scalar_le,
        P.scalar_log,
        P.scalar_lt,
        P.scalar_max,
        P.scalar_mod,
        P.scalar_mul,
        P.scalar_ne,
        P.scalar_pow,
        P.scalar_sign,
        P.scalar_sin,
        P.scalar_sub,
        P.scalar_tan,
        P.scalar_tanh,
        P.scalar_to_array,
        P.scalar_trunc,
        P.scalar_uadd,
        P.scalar_usub,
    ],
}


def has_computation(name):
    """Return True if given name is a computation."""
    return name in _COMPUTATIONS


def get_computation(name):
    """Return tuple of primitives associated to given computation."""
    return tuple(_COMPUTATIONS[name])


# Ignored primitives:
# P.array_scan,
# P.broadcast_shape,
# P.env_add,
# P.invert_permutation,
# P.partial,
# P.return_,
