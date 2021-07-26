"""Subset of Numpy symbols.

Allow to monitor which numpy symbols are currently used in testing.

Also allow to redfine some numpy functions without default argument values,
as this seems to trigger errors in myia parser.
"""

import numpy as np

uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64
uint64 = np.uint64
float16 = np.float16
float32 = np.float32
float64 = np.float64
dtype = np.dtype
ndarray = np.ndarray


def array(scalar):
    """Numpy array."""
    return np.array(scalar)


def broadcast_to(arr, shp):
    """Numpy broadcast_to."""
    return np.broadcast_to(arr, shp)


def dot(a, b):
    """Numpy dot."""
    return np.dot(a, b)


def full(shape, value, dtype):
    """Numpy full."""
    return np.full(shape, value, dtype)


def log(x):
    """Numpy log."""
    return np.log(x)


def ones(shape):
    """Numpy ones."""
    return np.ones(shape)


def prod(arr):
    """Numpy prod."""
    return np.prod(arr)


def reshape(arr, shp):
    """Numpy reshape."""
    return np.reshape(arr, shp)


def transpose(arr, perm):
    """Numpy transpose."""
    return np.transpose(arr, perm)


def vectorize(fn):
    """Numpy vectorize."""
    return np.vectorize(fn)
