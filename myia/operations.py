"""Myia operations."""


from operator import (  # noqa
    add, sub, mul, truediv, floordiv, mod, pow, eq, ne, lt, gt, le, ge,
    pos, neg, not_, and_, or_, matmul, getitem, setitem
)

from numpy import (  # noqa
    exp, log, sin, cos, tan
)

from builtins import (  # noqa
    bool, getattr, setattr, len
)


def make_tuple(*elts):  # pragma: no cover
    """Tuple builder."""
    raise RuntimeError('This operation is not meant to be called directly.')


def make_list(*elts):  # pragma: no cover
    """List builder."""
    raise RuntimeError('This operation is not meant to be called directly.')


def switch(cond, tb, fb):  # pragma: no cover
    """Switch statement, returns one of the two values."""
    raise RuntimeError('This operation is not meant to be called directly.')

def cast(x,t):
    raise RuntimeError('This operation is not meant to be called directly.')

def iter(xs):  # pragma: no cover
    """Myia iterator function."""
    raise RuntimeError('This operation is not meant to be called directly.')


def next(it):  # pragma: no cover
    """Myia next function."""
    raise RuntimeError('This operation is not meant to be called directly.')


def hasnext(it):  # pragma: no cover
    """Myia hasnext function."""
    raise RuntimeError('This operation is not meant to be called directly.')


def to_array(x):  # pragma: no cover
    """Myia to_array function."""
    raise RuntimeError('This operation is not meant to be called directly.')
