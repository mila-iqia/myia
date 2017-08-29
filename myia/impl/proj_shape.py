
from .proj import proj
from ..symbols import builtins


_ = True


###################
# Shape inferrers #
###################


@proj(builtins.shape)
def proj_add(x, y):
    assert shape(x) == shape(y)
    return shape(x)


@proj(builtins.shape)
def proj_dot(x, y):
    a, b = shape(x)
    c, d = shape(y)
    assert b == c
    return a, d
