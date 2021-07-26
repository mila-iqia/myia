import numpy as np


int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64
uint64 = np.uint64
float16 = np.float16
float32 = np.float32
dtype = np.dtype


def prod(arr):
    return np.prod(arr)


def full(shape, value, dtype):
    return np.full(shape, value, dtype)


def log(x):
    return np.log(x)


def ones(shape):
    return np.ones(shape)
