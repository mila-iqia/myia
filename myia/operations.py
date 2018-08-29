"""Myia operations."""


from operator import (  # noqa
    add, sub, mul, truediv, mod, pow, eq, ne, lt, gt, le, ge,
    pos, neg, not_, and_, or_, matmul, getitem, setitem
)

from math import (  # noqa
    exp, log, sin, cos, tan
)

from builtins import (  # noqa
    bool, getattr, setattr, len
)


def make_tuple(*elts):  # pragma: no cover
    """Tuple builder."""
    raise RuntimeError('This operation is not meant to be called directly.')


def cons_tuple(head, tail):  # pragma: no cover
    """Tuple builder."""
    raise RuntimeError('This operation is not meant to be called directly.')


def if_(cond, tb, fb):  # pragma: no cover
    """If statement, takes condition and two thunks."""
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
