"""Placeholder for master branch operations."""


def broadcast_shape(shpx, shpy):
    """Placeholder for master operation `broadcast_shape`."""
    raise NotImplementedError()


def dict_setitem(dct, k, v):
    """Placeholder for master operation `dict_setitem`."""
    raise NotImplementedError()


def dict_values(dct):
    """Placeholder for master operation `dict_values`."""
    raise NotImplementedError()


def embed(x):
    """Placeholder for master operation `embed`."""
    raise NotImplementedError()


def env_getitem(obj, k, default):
    """Placeholder for master operation `env_getitem`."""
    raise NotImplementedError()


def env_setitem(obj, k, v):
    """Placeholder for master operation `env_setitem`."""
    raise NotImplementedError()


def gadd(x, y):
    """Placeholder for master operation `gadd`."""
    raise NotImplementedError()


def grad(fn):
    """Placeholder for master operation `grad`."""
    raise NotImplementedError()


def hastype(obj, typ):
    """Placeholder for master operation `hastype`."""
    return isinstance(obj, typ)


def identity(x):
    """Placeholder for master operation `identity`."""
    return x


def J(fn):
    """Placeholder for master operation `J`."""
    raise NotImplementedError()


def Jinv(x):
    """Placeholder for master operation `Jinv`."""
    raise NotImplementedError()


def make_record(cls, *args):
    """Placeholder for master operation `make_record`."""
    raise NotImplementedError()


def record_setitem(obj, k, v):
    """Placeholder for master operation `record_setitem`."""
    raise NotImplementedError()


def scalar_add(a, b):
    """Placeholder for master operation `scalar_add`."""
    raise a + b


def scalar_cast(x, typ):
    """Placeholder for master operation `scalar_cast`."""
    return typ(x)


def scalar_lt(x, y):
    """Placeholder for master operation `scalar_lt`."""
    return x < y


def scalar_mul(x, y):
    """Placeholder for master operation `scalar_mul`."""
    return x * y


def scalar_usub(x):
    """Placeholder for master operation `scalar_usub`."""
    return -x


def tagged(x, tag=None):
    """Placeholder for master operation `tagged`."""
    raise NotImplementedError()


def tuple_setitem(t, idx, v):
    """Placeholder for master operation `tuple_setitem`."""
    o = list(t)
    o[idx] = v
    return tuple(o)


def unsafe_static_cast(x, typ):
    """Placeholder for master operation `unsafe_static_cast`."""
    return typ(x)


def zeros_like(x):
    """Placeholder for master operation `zeros_like`."""
    raise NotImplementedError()
