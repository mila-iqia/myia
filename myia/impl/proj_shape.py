
from .proj import proj, natproj
from .impl_abstract import abstract_shape as shape
from ..symbols import builtins
from ..inference.avm import WrappedException


_ = True


###################
# Shape inferrers #
###################


@proj(builtins.shape)
def proj_identity(x):
    return shape(x)


@natproj(builtins.shape)
def proj_mktuple(*args):
    raise WrappedException(Exception('Tuples have no shape.'))


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


@proj(builtins.shape)
def proj_greater(x, y):
    raise Exception('No shape (greater).')


@proj(builtins.shape)
def proj_less(x, y):
    raise Exception('No shape (less).')


@proj(builtins.shape)
def proj_switch(cond, x, y):
    raise Exception('No shape (switch).')
