"""Myia operations."""


from builtins import bool, getattr, len, setattr  # noqa
from operator import (  # noqa
    add,
    and_,
    eq,
    floordiv,
    ge,
    getitem,
    gt,
    is_,
    is_not,
    le,
    lt,
    matmul,
    mod,
    mul,
    ne,
    neg,
    not_,
    or_,
    pos,
    pow,
    setitem,
    sub,
    truediv,
)

from numpy import cos, exp, log, sin, tan  # noqa


def make_tuple(*elts):  # pragma: no cover
    """Tuple builder."""
    raise RuntimeError('This operation is not meant to be called directly.')


def make_list(*elts):  # pragma: no cover
    """List builder."""
    raise RuntimeError('This operation is not meant to be called directly.')


def make_dict(k, v):  # pragma: no cover
    """Dict builder."""
    raise RuntimeError('This operation is not meant to be called directly.')


def switch(cond, tb, fb):  # pragma: no cover
    """Switch statement, returns one of the two values."""
    raise RuntimeError('This operation is not meant to be called directly.')


def user_switch(cond, tb, fb):  # pragma: no cover
    """Switch statement, returns one of the two values."""
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


def to_array(x, t):  # pragma: no cover
    """Myia to_array function."""
    raise RuntimeError('This operation is not meant to be called directly.')


def slice(start, stop, step):  # pragma: no cover
    """Slice function."""
    raise RuntimeError('This operation is not meant to be called directly.')


def apply(fn, *arg_groups):  # pragma: no cover
    """Function application."""
    raise RuntimeError('This operation is not meant to be called directly.')
