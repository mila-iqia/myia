import numpy as np
from myia.infer.inferrers import X, signature, getattr_inferrer, AbstractNone
from myia.infer.infnode import inferrers, inference_function, InferenceEngine
from myia.testing.common import Number, Float, array_of, Ty, tuple_of, Nil
from myia.infer.algo import Require
from myia.abstract import data
from myia.testing import numpy_subset
from myia.abstract.map import MapError


def array_cast(arr, typ): raise NotImplementedError()
def array_map(fn, arr, *arrays): raise NotImplementedError()
def array_reduce(fn, arr, shap): raise NotImplementedError()
def array_to_scalar(x): raise NotImplementedError()
def bool_and(a, b): raise NotImplementedError()
def bool_or(a, b): raise NotImplementedError()
def broadcast_shape(shpx, shpy): raise NotImplementedError()
def conv2d(inp, weights, stride, padding, dilation, groups): raise NotImplementedError()
def conv2d_grad_input(inp_size, weights, grad_output, stride, padding, dilation, groups): raise NotImplementedError()
def conv2d_weight_grad(inp, weight_size, grad_output, stride, padding, dilation, groups): raise NotImplementedError()
def dict_setitem(dct, k, v): raise NotImplementedError()
def dict_values(dct): raise NotImplementedError()
def distribute(arr, shp): raise NotImplementedError()
def dot(a, b): raise NotImplementedError()
def embed(x): raise NotImplementedError()
def env_getitem(obj, k, default): raise NotImplementedError()
def env_setitem(obj, k, v): raise NotImplementedError()
def gadd(x, y): raise NotImplementedError()
def grad(fn): raise NotImplementedError()
def hastype(obj, typ): raise NotImplementedError()
def identity(x): raise NotImplementedError()
def J(fn): raise NotImplementedError()
def Jinv(x): raise NotImplementedError()
def make_record(cls, *args): raise NotImplementedError()
def nil_eq(a, b): raise NotImplementedError()
def nil_ne(a, b): raise NotImplementedError()
def Operation(): raise NotImplementedError()
def partial(f, *args): raise NotImplementedError()
def record_setitem(obj, k, v): raise NotImplementedError()
def reshape(arr, shp): raise NotImplementedError()
def scalar_add(a, b): raise NotImplementedError()
def scalar_cast(x, typ): raise NotImplementedError()
def scalar_lt(): raise NotImplementedError()
def scalar_mul(x, y): return x * y
def scalar_to_array(x): return np.array(x)
def scalar_usub(): raise NotImplementedError()
def shape(arr): raise NotImplementedError()
def switch(c, t, f): raise NotImplementedError()
def tagged(x, tag=None): raise NotImplementedError()
def transpose(arr, perm): raise NotImplementedError()
def tuple_setitem(t, idx, v): raise NotImplementedError()
def typeof(obj): raise NotImplementedError()
def unsafe_static_cast(x, typ): raise NotImplementedError()
def user_switch(c, t, f): raise NotImplementedError()
def zeros_like(x): raise NotImplementedError()


def np_full_inferrer(node, args, unif):
    shape_node, value_node, dtype_node = args
    shape_type = yield Require(shape_node)
    value_type = yield Require(value_node)
    dtype = yield Require(dtype_node)
    if dtype is Nil:
        dtype = value_type
    elif isinstance(dtype, data.AbstractAtom) and dtype.tracks.interface is str:
        type_name = dtype.tracks.value
        try:
            dtype = data.AbstractAtom({"interface": np.dtype(type_name).type})
        except TypeError:
            raise TypeError(f"Cannot parse numpy dtype {type_name}")
    elif InferenceEngine.is_abstract_type(dtype):
        dtype = dtype.elements[0]
    else:
        raise TypeError(f"Expected an abstract type, got {dtype}")
    return data.AbstractStructure([dtype], {"interface": np.ndarray})


def add_testing_inferrers():
    inferrers.update({
        numpy_subset.log: signature(Number, ret=Float),
        np.array: signature(Number, ret=array_of(Number, ())),
        numpy_subset.prod: signature(array_of(Number), ret=array_of(Number, ())),
        # numpy_subset.full: signature(
        #     tuple_of(), Number, Ty(Number), ret=array_of(Number)
        # ),
        numpy_subset.full: inference_function(np_full_inferrer),
        type(None): signature(ret=Nil),
    })
