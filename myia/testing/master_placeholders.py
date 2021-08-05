"""Placeholder for master branch operations."""
from ovld import ovld


def dict_setitem(dct, k, v):
    """Placeholder for master operation `dict_setitem`."""
    dct[k] = v
    return dct


def dict_values(dct):
    """Placeholder for master operation `dict_values`."""
    return tuple(dct.values())


def hastype(obj, typ):
    """Placeholder for master operation `hastype`."""
    return isinstance(obj, typ)


def identity(x):
    """Placeholder for master operation `identity`."""
    return x


def scalar_add(a, b):
    """Placeholder for master operation `scalar_add`."""
    raise a + b


def scalar_lt(x, y):
    """Placeholder for master operation `scalar_lt`."""
    return x < y


def scalar_mul(x, y):
    """Placeholder for master operation `scalar_mul`."""
    return x * y


def scalar_usub(x):
    """Placeholder for master operation `scalar_usub`."""
    return -x


def tuple_setitem(t, idx, v):
    """Placeholder for master operation `tuple_setitem`."""
    o = list(t)
    o[idx] = v
    return tuple(o)


@ovld
def zeros_like(self, x: int):  # noqa: F811
    """Placeholder for master operation `zeros_like`."""
    return 0


@ovld
def zeros_like(self, x: float):  # noqa: F811
    return 0.0


@ovld
def zeros_like(self, x: list):  # noqa: F811
    return [self(v) for v in x]


@ovld
def zeros_like(self, x: tuple):  # noqa: F811
    return tuple(self(v) for v in x)
