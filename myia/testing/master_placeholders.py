"""Placeholder for master branch operations."""
from myia.testing import numpy_subset


def array_cast(arr, typ):
    """Placeholder for master operation `array_cast`."""
    return arr.astype(typ)


def array_map(fn, *arrays):
    """Placeholder for master operation `array_map`."""
    return numpy_subset.vectorize(fn)(*arrays)


def array_reduce(fn, arr, shap):
    """Placeholder for master operation `array_reduce`."""
    raise NotImplementedError()


def array_to_scalar(x):
    """Placeholder for master operation `array_to_scalar`."""
    return x.item()


def broadcast_shape(shpx, shpy):
    """Placeholder for master operation `broadcast_shape`."""
    raise NotImplementedError()


def conv2d(inp, weights, stride, padding, dilation, groups):
    """Placeholder for master operation `conv2d`."""
    raise NotImplementedError()


def conv2d_grad_input(
    inp_size, weights, grad_output, stride, padding, dilation, groups
):
    """Placeholder for master operation `conv2d_grad_input`."""
    raise NotImplementedError()


def conv2d_weight_grad(
    inp, weight_size, grad_output, stride, padding, dilation, groups
):
    """Placeholder for master operation `conv2d_weight_grad`."""
    raise NotImplementedError()


def dict_setitem(dct, k, v):
    """Placeholder for master operation `dict_setitem`."""
    raise NotImplementedError()


def dict_values(dct):
    """Placeholder for master operation `dict_values`."""
    raise NotImplementedError()


def distribute(arr, shp):
    """Placeholder for master operation `distribute`."""
    return numpy_subset.broadcast_to(arr, shp)


def dot(a, b):
    """Placeholder for master operation `dot`."""
    return numpy_subset.dot(a, b)


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


def reshape(arr, shp):
    """Placeholder for master operation `reshape`."""
    return numpy_subset.reshape(arr, shp)


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


def scalar_to_array(x):
    """Placeholder for master operation `scalar_to_array`."""
    return numpy_subset.array(x)


def scalar_usub(x):
    """Placeholder for master operation `scalar_usub`."""
    return -x


def shape(arr):
    """Placeholder for master operation `shape`."""
    return arr.shape


def tagged(x, tag=None):
    """Placeholder for master operation `tagged`."""
    raise NotImplementedError()


def transpose(arr, perm):
    """Placeholder for master operation `transpose`."""
    return numpy_subset.transpose(arr, perm)


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
