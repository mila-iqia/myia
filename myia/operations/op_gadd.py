"""Implementation of the 'gadd' operation."""

from ..lib import AbstractRandomState, HyperMap, MultitypeGraph, core
from ..xtype import Bool, EnvType, Nil, Number
from . import zeros_like
from .primitives import bool_or, env_add, scalar_add

_leaf_add = MultitypeGraph("gadd")


@_leaf_add.register(Number, Number)
@core
def _scalar_add(x, y):
    return scalar_add(x, y)


@_leaf_add.register(EnvType, EnvType)
@core
def _sm_add(x, y):
    return env_add(x, y)


@_leaf_add.register(Nil, Nil)
@core
def _nil_add(x, y):
    return None


@_leaf_add.register(Bool, Bool)
@core
def _bool_add(x, y):
    return bool_or(x, y)


@_leaf_add.register(AbstractRandomState, AbstractRandomState)
@core
def _rstate_add(x, y):
    """Just return zeros_like by default."""
    return zeros_like(x)


gadd = HyperMap(
    name="gadd", fn_leaf=_leaf_add, broadcast=False, trust_union_match=True
)


__operation_defaults__ = {
    "name": "gadd",
    "registered_name": "gadd",
    "mapping": gadd,
    "python_implementation": None,
}
