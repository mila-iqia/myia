
from .proj import proj
from ..symbols import builtins


_ = True


##################
# Type inferrers #
##################


@proj(builtins.type)
def proj_add(x, y):
    tx = type(x)
    ty = type(y)
    #assert tx.name == 'Array'
    assert tx == ty
    return tx


@proj(builtins.type)
def proj_dot(x, y):
    tx = type(x)
    ty = type(y)
    #assert tx.name == 'Array'
    assert tx == ty
    return tx
