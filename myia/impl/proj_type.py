
from .proj import natproj
from ..symbols import builtins
from ..inference.avm import AbstractValue, WrappedException
from ..inference.types import typeof


_ = True


def getprop(v, sym):
    if isinstance(v, AbstractValue):
        return v[sym]
    else:
        raise Exception(v)


##################
# Type inferrers #
##################


@natproj(builtins.type)
def proj_add(x, y):
    # tx = typeof(x)
    # ty = typeof(y)
    tx = getprop(x, builtins.type)
    ty = getprop(y, builtins.type)
    assert tx.name == 'Array'
    if tx == ty:
        return tx
    else:
        raise WrappedException(TypeError('Type error (add).'))


# @proj(builtins.type)
# def proj_dot(x, y):
#     tx = type(x)
#     ty = type(y)
#     #assert tx.name == 'Array'
#     if tx == ty:
#         return tx
#     else:
#         raise Exception('Type error (dot).')
