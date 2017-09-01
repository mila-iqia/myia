
from .proj import proj
from ..symbols import builtins


_ = True


###################
# Shape inferrers #
###################


@proj(builtins.shape)
def proj_identity(x):
    return shape(x)


@proj(builtins.shape)
def proj_add(x, y):
    if shape(x) == shape(y):
        return shape(x)
    else:
        raise Exception('Shape error (add).')


@proj(builtins.shape)
def proj_subtract(x, y):
    if shape(x) == shape(y):
        return shape(x)
    else:
        raise Exception('Shape error (subtract).')


@proj(builtins.shape)
def proj_dot(x, y):
    a, b = shape(x)
    c, d = shape(y)
    if b == c:
        return a, d
    else:
        raise Exception('Shape error (dot).')
