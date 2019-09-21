"""Implementation of the 'gadd' operation."""

from ..lib import HyperMap, MultitypeGraph, core
from ..xtype import EnvType, Nil, Number
from .primitives import env_add, scalar_add

_leaf_add = MultitypeGraph('gadd')


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


gadd = HyperMap(name='gadd', fn_leaf=_leaf_add,
                broadcast=False, trust_union_match=True)


__operation_defaults__ = {
    'name': 'gadd',
    'registered_name': 'gadd',
    'mapping': gadd,
    'python_implementation': None,
}
