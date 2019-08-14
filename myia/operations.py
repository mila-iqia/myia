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


class MyiaOnlyOperation:
    """Represents a Myia-only operation."""

    def __init__(self, name):
        """Initialize a MyiaOnlyOperation."""
        self.name = name

    def __call__(self):  # pragma: no cover
        """Call a MyiaOnlyOperation."""
        raise RuntimeError(
            f'Myia-only operation {self.name} cannot be called directly.'
        )

    def __str__(self):
        return f'myia.operations.{self.name}'

    __repr__ = __str__


apply = MyiaOnlyOperation('apply')
hasnext = MyiaOnlyOperation('hasnext')
iter = MyiaOnlyOperation('iter')
make_tuple = MyiaOnlyOperation('make_tuple')
make_list = MyiaOnlyOperation('make_list')
make_dict = MyiaOnlyOperation('make_dict')
next = MyiaOnlyOperation('next')
slice = MyiaOnlyOperation('slice')
switch = MyiaOnlyOperation('switch')
to_array = MyiaOnlyOperation('to_array')
user_switch = MyiaOnlyOperation('user_switch')
