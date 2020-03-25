"""Scalar operations."""

from ..lib import core
from ..operations import floor, hastype, scalar_cast, scalar_div, typeof
from ..xtype import f32, f64, i8, i16, u8, u16
from .utils import to_opdef


@to_opdef
@core(static_inline=True)
def int_floordiv(x, y):
    """Implementation of `int_floordiv`."""
    if (x <= 0) == (y <= 0):
        return scalar_div(x, y)
    else:
        return scalar_div(x, y) - 1


@to_opdef
@core(static_inline="inner")
def int_truediv(x, y):
    """Implementation of `int_truediv`."""
    if hastype(x, typeof(y)):
        if (
            hastype(x, i8)
            or hastype(x, u8)
            or hastype(x, i16)
            or hastype(x, u16)
        ):
            return scalar_div(scalar_cast(x, f32), scalar_cast(y, f32))
        return scalar_div(scalar_cast(x, f64), scalar_cast(y, f64))
    else:
        raise Exception("Incompatible types for division.")


@to_opdef
@core(static_inline=True)
def float_floordiv(x, y):
    """Implementation of `float_floordiv`."""
    return floor(x / y)


@to_opdef
@core(static_inline=True)
def int_bool(x):
    """Implementation of `int_bool`."""
    return x != 0


@to_opdef
@core(static_inline=True)
def float_bool(x):
    """Implementation of `float_bool`."""
    return x != 0.0
