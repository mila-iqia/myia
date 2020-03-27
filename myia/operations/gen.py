"""Utilities to generate methods."""

from ..lib import core
from . import hastype


def lop(op, type, name):
    """Generate a left hand side method.

    Returns NotImplemented if the second input has the wrong type.
    """

    @core(name=name, static_inline=True)
    def protocol(x, y):
        if hastype(y, type):
            return op(x, y)
        else:
            return NotImplemented

    return protocol


def rop(op, type, name):
    """Generate a right hand side method.

    Returns NotImplemented if the second input has the wrong type.
    """

    @core(name=name, static_inline=True)
    def protocol(x, y):
        if hastype(y, type):
            return op(y, x)
        else:
            return NotImplemented

    return protocol


def reverse_binop(op, name):
    """Reverse the argument order of a binary function."""

    @core(name=name, static_inline=True)
    def protocol(x, y):
        return op(y, x)

    return protocol
