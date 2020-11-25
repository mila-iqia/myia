"""Definitions of computations.

A computation is a group of backend primitives that perform related operations.
To support a computation, a backend must implement all related primitives.
"""
from myia.operations import primitives as P
from myia.operations.utils import Primitive


class PrimGroup:
    """Group of primitives that perform related operations.

    Defined by a name and a list of primitives. May be also used
    to represent a unique primitive, with name=None and primitives
    list containing only 1 primitive.
    """

    def __init__(self, name, primitives):
        """Initialize."""
        self.name = name  # type: str
        self.primitives = primitives  # type: list

    def __str__(self):
        if self.name is None and len(self.primitives) == 1:
            return f"Primitive({self.primitives[0].name})"
        return f"PrimGroup[{self.name}]({', '.join(p.name for p in self.primitives)})"

    @classmethod
    def ensure(cls, prim_or_group):
        """Make sure given object is a primitive or a group of primitives.

        Convert a primitive to a PrimGroup if necessary.

        :return a valid PrimGroup object
        """
        if isinstance(prim_or_group, PrimGroup):
            return prim_or_group
        assert isinstance(prim_or_group, Primitive)
        return cls(None, [prim_or_group])


concat_split_operations = PrimGroup(
    "concat_split_operations", [P.concat, P.split]
)
conv2d_operations = PrimGroup(
    "conv2d_operations", [P.conv2d, P.conv2d_weight_grad, P.conv_transpose2d]
)
max_pool2d_operations = PrimGroup(
    "max_pool2d_operations", [P.max_pool2d, P.max_pool2d_grad]
)
gather_scatter_operations = PrimGroup(
    "gather_scatter_operations", [P.gather, P.scatter, P.scatter_add]
)
take_operations = PrimGroup("take_operations", [P.take, P.take_grad_inp])
exception_operations = PrimGroup(
    "exception_operations", [P.raise_, P.make_exception]
)
tag_operations = PrimGroup("tag_operations", [P.casttag, P.tagged, P.hastag])
env_operations = PrimGroup("env_operations", [P.env_getitem, P.env_setitem])
tuple_operations = PrimGroup(
    "tuple_operations", [P.make_tuple, P.tuple_getitem, P.tuple_setitem]
)
universe_operations = PrimGroup(
    "universe_operations",
    [P.make_handle, P.universe_getitem, P.universe_setitem],
)
rng_operations = PrimGroup(
    "rng_operations", [P.random_initialize, P.random_uint32]
)
logical_operations = PrimGroup(
    "logical_operations",
    [P.bool_and, P.bool_eq, P.bool_not, P.bool_or, P.switch],
)
array_operations = PrimGroup(
    "array_operations",
    [
        P.argmax,
        P.array_max,
        P.array_map,
        P.array_reduce,
        P.array_getitem,
        P.array_setitem,
        P.array_to_scalar,
        P.distribute,
        P.dot,
        P.reshape,
        P.scalar_to_array,
        P.shape,
        P.transpose,
        P.array_cast,
    ],
)
bitwise_operations = PrimGroup(
    "bitwise_operations",
    [
        P.scalar_bit_and,
        P.scalar_bit_lshift,
        P.scalar_bit_not,
        P.scalar_bit_or,
        P.scalar_bit_rshift,
        P.scalar_bit_xor,
    ],
)
scalar_operations = PrimGroup(
    "scalar_operations",
    [
        P.scalar_abs,
        P.scalar_add,
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
        P.scalar_trunc,
        P.scalar_uadd,
        P.scalar_usub,
    ],
)


# Ignored primitives:
# P.unsafe_static_cast
# P.array_scan
# P.broadcast_shape
# P.env_add
# P.invert_permutation
# P.partial
# P.return_
