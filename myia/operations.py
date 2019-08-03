"""Myia operations."""


from operator import (  # noqa
    add, sub, mul, truediv, floordiv, mod, pow, eq, ne, lt, gt, le, ge,
    pos, neg, not_, and_, or_, matmul, getitem, setitem, is_, is_not
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
