"""Implementation of the 'zeros_like' operation."""

from ..lib import (
    ANYTHING,
    AbstractArray,
    AbstractClassBase,
    AbstractDict,
    AbstractError,
    AbstractFunction,
    AbstractRandomState,
    AbstractTaggedUnion,
    AbstractTuple,
    AbstractUnion,
    HyperMap,
    MultitypeGraph,
    core,
    newenv,
)
from ..operations import myia_to_array, typeof
from ..xtype import Bool, EnvType, Nil, Number, SymbolicKeyType
from .primitives import distribute, scalar_cast, shape

_leaf_zeros_like = MultitypeGraph("zeros_like")


@_leaf_zeros_like.register(AbstractFunction(value=ANYTHING))
@core
def _function_zero(_):
    return newenv


@_leaf_zeros_like.register(AbstractError(ANYTHING))
@core
def _dead_zero(x):
    return x


@_leaf_zeros_like.register(Bool)
@core
def _bool_zero(_):
    return False


@_leaf_zeros_like.register(Nil)
@core
def _nil_zero(_):
    return None


@_leaf_zeros_like.register(Number)
@core
def _scalar_zero(x):
    return scalar_cast(0, typeof(x))


@_leaf_zeros_like.register(EnvType)
@core
def _env_type_zero(x):
    return newenv


@_leaf_zeros_like.register(SymbolicKeyType)
@core
def _symbolic_key_type_zero(x):
    return x


@_leaf_zeros_like.register(AbstractArray)
@core
def _array_zero(xs):
    scalar_zero = scalar_cast(0, typeof(xs).element)
    return distribute(myia_to_array(scalar_zero, typeof(xs)), shape(xs))


@_leaf_zeros_like.register(AbstractRandomState)
@core
def _rng_zero(x):
    """Return zero scalar, according to sensitivity_transform."""
    return 0


zeros_like = HyperMap(
    name="zeros_like",
    nonleaf=(
        AbstractTuple,
        AbstractClassBase,
        AbstractUnion,
        AbstractTaggedUnion,
        AbstractDict,
    ),
    fn_leaf=_leaf_zeros_like,
)


__operation_defaults__ = {
    "name": "zeros_like",
    "registered_name": "zeros_like",
    "mapping": zeros_like,
    "python_implementation": None,
}
