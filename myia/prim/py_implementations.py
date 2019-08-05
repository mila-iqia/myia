"""Implementations for the debug VM."""

from copy import copy
from functools import reduce
from typing import Callable
import numpy as np
import math

from .. import dtype as types, abstract
from ..dtype import Number, Float, Bool, pytype_to_myiatype
from ..utils import Registry, TaggedValue

from . import ops as primops


py_registry: Registry[primops.Primitive, Callable] = Registry()
vm_registry: Registry[primops.Primitive, Callable] = Registry()
py_register = py_registry.register
vm_register = vm_registry.register


def register(prim):
    """Register an implementation for this primitive.

    The same implementation will be used for both the VM and for the pure
    Python version.
    """
    def deco(fn):
        vm_register(prim)(lambda vm, *args: fn(*args))
        return py_register(prim)(fn)
    return deco


def _assert_scalar(*args):
    # TODO: These checks should be stricter, e.g. require that all args
    # have exactly the same type, but right now there is some mixing between
    # numpy types and int/float.
    for x in args:
        if isinstance(x, np.ndarray):
            if x.shape != ():
                msg = f'Expected scalar, not array with shape {x.shape}'
                raise TypeError(msg)
        elif not isinstance(x, (int, float, np.number)):
            raise TypeError(f'Expected scalar, not {type(x)}')


@register(primops.scalar_add)
def scalar_add(x: Number, y: Number) -> Number:
    """Implement `scalar_add`."""
    _assert_scalar(x, y)
    return x + y


@register(primops.scalar_sub)
def scalar_sub(x: Number, y: Number) -> Number:
    """Implement `scalar_sub`."""
    _assert_scalar(x, y)
    return x - y


@register(primops.scalar_mul)
def scalar_mul(x: Number, y: Number) -> Number:
    """Implement `scalar_mul`."""
    _assert_scalar(x, y)
    return x * y


@register(primops.scalar_div)
def scalar_div(x: Number, y: Number) -> Number:
    """Implement `scalar_div`."""
    _assert_scalar(x, y)
    if isinstance(x, (float, np.floating)):
        return x / y
    else:
        return int(x / y)


@register(primops.scalar_mod)
def scalar_mod(x: Number, y: Number) -> Number:
    """Implement `scalar_mod`."""
    _assert_scalar(x, y)
    return x % y


@register(primops.scalar_pow)
def scalar_pow(x: Number, y: Number) -> Number:
    """Implement `scalar_pow`."""
    _assert_scalar(x, y)
    return x ** y


@register(primops.scalar_trunc)
def scalar_trunc(x: Number) -> Number:
    """Implement `scalar_trunc`."""
    _assert_scalar(x)
    return np.trunc(x)


@register(primops.scalar_floor)
def scalar_floor(x: Number) -> Number:
    """Implement `scalar_floor`."""
    _assert_scalar(x)
    return np.floor(x)


@register(primops.scalar_uadd)
def scalar_uadd(x: Number) -> Number:
    """Implement `scalar_uadd`."""
    _assert_scalar(x)
    return x


@register(primops.scalar_usub)
def scalar_usub(x: Number) -> Number:
    """Implement `scalar_usub`."""
    _assert_scalar(x)
    return -x


@register(primops.scalar_exp)
def scalar_exp(x: Number) -> Number:
    """Implement `scalar_exp`."""
    _assert_scalar(x)
    return math.exp(x)


@register(primops.scalar_log)
def scalar_log(x: Float) -> Float:
    """Implement `scalar_log`."""
    _assert_scalar(x)
    return math.log(x)


@register(primops.scalar_sin)
def scalar_sin(x: Number) -> Number:
    """Implement `scalar_sin`."""
    _assert_scalar(x)
    return math.sin(x)


@register(primops.scalar_cos)
def scalar_cos(x: Number) -> Number:
    """Implement `scalar_cos`."""
    _assert_scalar(x)
    return math.cos(x)


@register(primops.scalar_tan)
def scalar_tan(x: Number) -> Number:
    """Implement `scalar_tan`."""
    _assert_scalar(x)
    return math.tan(x)


@register(primops.scalar_tanh)
def scalar_tanh(x: Number) -> Number:
    """Implement `scalar_tanh`."""
    _assert_scalar(x)
    return math.tanh(x)


@register(primops.scalar_eq)
def scalar_eq(x: Number, y: Number) -> Bool:
    """Implement `scalar_eq`."""
    _assert_scalar(x, y)
    return x == y


@register(primops.scalar_lt)
def scalar_lt(x: Number, y: Number) -> Bool:
    """Implement `scalar_lt`."""
    _assert_scalar(x, y)
    return x < y


@register(primops.scalar_gt)
def scalar_gt(x: Number, y: Number) -> Bool:
    """Implement `scalar_gt`."""
    _assert_scalar(x, y)
    return x > y


@register(primops.scalar_ne)
def scalar_ne(x: Number, y: Number) -> Bool:
    """Implement `scalar_ne`."""
    _assert_scalar(x, y)
    return x != y


@register(primops.scalar_le)
def scalar_le(x: Number, y: Number) -> Bool:
    """Implement `scalar_le`."""
    _assert_scalar(x, y)
    return x <= y


@register(primops.scalar_ge)
def scalar_ge(x: Number, y: Number) -> Bool:
    """Implement `scalar_ge`."""
    _assert_scalar(x, y)
    return x >= y


@register(primops.bool_not)
def bool_not(x: Bool) -> Bool:
    """Implement `bool_not`."""
    assert x is True or x is False
    return not x


@register(primops.bool_and)
def bool_and(x: Bool, y: Bool) -> Bool:
    """Implement `bool_and`."""
    assert x is True or x is False
    assert y is True or y is False
    return x and y


@register(primops.bool_or)
def bool_or(x: Bool, y: Bool) -> Bool:
    """Implement `bool_or`."""
    assert x is True or x is False
    assert y is True or y is False
    return x or y


@register(primops.bool_eq)
def bool_eq(x: Bool, y: Bool) -> Bool:
    """Implement `bool_eq`."""
    assert x is True or x is False
    assert y is True or y is False
    return x == y


@register(primops.typeof)
def typeof(x):
    """Implement typeof."""
    return abstract.from_value(x, broaden=True)


@register(primops.hastype)
def hastype(x, t):
    """Implement `hastype`."""
    from ..abstract import type_to_abstract, TYPE, AbstractScalar, \
        AbstractClassBase, ANYTHING
    tt = type_to_abstract(t)
    if isinstance(tt, AbstractScalar):
        try:
            typ = pytype_to_myiatype(type(x))
            return issubclass(typ, tt.values[TYPE])
        except KeyError:
            return False
    elif (isinstance(tt, AbstractClassBase)
          and all(v is ANYTHING for k, v in tt.attributes.items())):
        return isinstance(x, tt.tag)
    else:
        raise NotImplementedError(
            'The Python implementation of hastype only support simple types.'
        )
        # from ..abstract import hastype_helper, ANYTHING
        # v = hastype_helper(typeof(x), type_to_abstract(t))
        # if v is ANYTHING:
        #     raise AssertionError()
        # return v


@register(primops.make_tuple)
def make_tuple(*args):
    """Implement `make_tuple`."""
    return args


@py_register(primops.tuple_getitem)
def tuple_getitem(data, item):
    """Implement `getitem`."""
    return data[item]


@py_register(primops.list_getitem)
def list_getitem(data, item):
    """Implement `getitem`."""
    return data[item]


@py_register(primops.dict_getitem)
def dict_getitem(data, item):
    """Implement `dict_getitem`."""
    return data[item]


@py_register(primops.array_getitem)
def array_getitem(data, item):
    """Implement `getitem`."""
    return data[item]


@vm_register(primops.tuple_getitem)
@vm_register(primops.list_getitem)
@vm_register(primops.array_getitem)
def _vm_getitem(vm, data, item):
    """Implement `getitem`."""
    return vm.convert(data[item])


@register(primops.tuple_setitem)
def tuple_setitem(data, item, value):
    """Implement `tuple_setitem`."""
    return tuple(value if i == item else x
                 for i, x in enumerate(data))


@register(primops.list_setitem)
def list_setitem(data, item, value):
    """Implement `list/array_setitem`."""
    data2 = copy(data)
    data2[item] = value
    return data2


@register(primops.array_setitem)
def array_setitem(data, item, value):
    """Implement `list/array_setitem`."""
    data2 = copy(data)
    data2[item] = value
    return data2


@register(primops.list_append)
def list_append(data, value):
    """Implement `list_append`."""
    data2 = copy(data)
    data2.append(value)
    return data2


@vm_register(primops.getattr)
def _vm_getattr(vm, data, attr):
    """Implement `getattr`."""
    from types import MethodType, BuiltinMethodType
    from ..vm import Partial
    from ..abstract import type_token
    # I don't know how else to get a reference to this type
    method_wrapper_type = type((0).__add__)
    try:
        x = getattr(data, attr)
    except AttributeError:
        t = type_token(typeof(data))
        mmap = vm.convert.resources.method_map[t]
        if attr in mmap:
            return Partial(vm.convert(mmap[attr]), [data], vm)
        else:
            raise  # pragma: no cover
    if isinstance(x, method_wrapper_type):
        # This is returned by <int>.__add__ and the like.
        # Don't know how else to retrieve the unwrapped method
        unwrapped = getattr(x.__objclass__, x.__name__)
        return Partial(vm.convert(unwrapped), [x.__self__], vm)
    elif isinstance(x, BuiltinMethodType) and hasattr(type(data), attr):
        # This is returned by <list>.__getitem__ and maybe others.
        x = getattr(type(data), attr)
        return Partial(vm.convert(x), [data], vm)
    elif isinstance(x, MethodType):
        # This is a method made from a user function
        return Partial(vm.convert(x.__func__), [x.__self__], vm)
    else:
        return vm.convert(x)


py_setattr = setattr


@register(primops.setattr)
def setattr(data, attr, value):
    """Implement `setattr`."""
    data2 = copy(data)
    py_setattr(data2, attr, value)
    return data2


@register(primops.shape)
def shape(array):
    """Implement `shape`."""
    return array.shape


@py_register(primops.array_map)
def array_map(fn, *arrays):
    """Implement `array_map`."""
    return np.vectorize(fn)(*arrays)


@vm_register(primops.array_map)
def _array_map_vm(vm, fn, *arrays):
    def fn_(*args):
        return vm.call(fn, args)
    return array_map(fn_, *arrays)


@py_register(primops.array_scan)
def array_scan(fn, init, array, axis):
    """Implement `array_scan`."""
    # This is inclusive scan because it's easier to implement
    # We will have to discuss what semantics we want later
    def f(ary):
        val = init
        it = np.nditer([ary, None])
        for x, y in it:
            val = fn(val, x)
            y[...] = val
        return it.operands[1]
    return np.apply_along_axis(f, axis, array)


@vm_register(primops.array_scan)
def _array_scan_vm(vm, fn, init, array, axis):
    def fn_(a, b):
        return vm.call(fn, [a, b])
    return array_scan(fn_, init, array, axis)


@py_register(primops.array_reduce)
def array_reduce(fn, array, shp):
    """Implement `array_reduce`."""
    idtype = array.dtype
    ufn = np.frompyfunc(fn, 2, 1)
    delta = len(array.shape) - len(shp)
    if delta < 0:
        raise ValueError('Shape to reduce to cannot be larger than original')

    def is_reduction(ishp, tshp):
        if tshp == 1 and ishp > 1:
            return True
        elif tshp != ishp:
            raise ValueError('Dimension mismatch for reduce')
        else:
            return False

    reduction = [(delta + idx if is_reduction(ishp, tshp) else None, True)
                 for idx, (ishp, tshp)
                 in enumerate(zip(array.shape[delta:], shp))]

    reduction = [(i, False) for i in range(delta)] + reduction

    for idx, keep in reversed(reduction):
        if idx is not None:
            array = ufn.reduce(array, axis=idx, keepdims=keep)

    if not isinstance(array, np.ndarray):
        # Force result to be ndarray, even if it's 0d
        array = np.array(array)

    array = array.astype(idtype)

    return array


@vm_register(primops.array_reduce)
def _array_reduce_vm(vm, fn, array, shp):
    def fn_(a, b):
        return vm.call(fn, [a, b])
    return array_reduce(fn_, array, shp)


@register(primops.distribute)
def distribute(v, shape):
    """Implement `distribute`."""
    return np.broadcast_to(v, shape)


@register(primops.reshape)
def reshape(v, shape):
    """Implement `reshape`."""
    return np.reshape(v, shape)


@register(primops.transpose)
def transpose(v, permutation):
    """Implement `transpose`."""
    return np.transpose(v, permutation)


@register(primops.dot)
def dot(a, b):
    """Implement `dot`."""
    return np.dot(a, b)


@register(primops.return_)
def return_(x):
    """Implement `return_`."""
    return x


@register(primops.raise_)
def raise_(x):
    """Implement `raise_`."""
    raise x


@register(primops.exception)
def exception(x):
    """Implement `exception`."""
    return Exception(x)


@register(primops.scalar_cast)
def scalar_cast(x, t):
    """Implement `scalar_cast`."""
    from ..abstract import type_to_abstract
    t = type_to_abstract(t)
    assert isinstance(t, abstract.AbstractScalar)
    t = t.values[abstract.TYPE]
    assert issubclass(t, types.Number)
    dtype = types.type_to_np_dtype(t)
    return getattr(np, dtype)(x)


@py_register(primops.list_map)
def list_map(f, *lsts):
    """Implement `list_map` in pure Python."""
    assert len(set(len(l) for l in lsts)) == 1

    def f_(args):
        return f(*args)
    return list(map(f_, zip(*lsts)))


@vm_register(primops.list_map)
def _list_map_vm(vm, f, *lsts):
    """Implement `list_map` for Myia's VM."""
    assert len(set(len(l) for l in lsts)) == 1

    def f_(args):
        return vm.call(f, args)
    return list(map(f_, zip(*lsts)))


@register(primops.identity)
def identity(x):
    """Implement `identity`."""
    return x


@vm_register(primops.resolve)
def _resolve_vm(vm, data, item):
    """Implement `resolve` for the VM."""
    # There is no Python implementation for this one.
    value = data[item]
    return vm.convert(value)


@py_register(primops.partial)
def partial(f, *args):
    """Implement `partial`."""
    def res(*others):
        return f(*(args + others))
    return res


@register(primops.switch)
def switch(c, x, y):
    """Implement `switch`."""
    return x if c else y


@register(primops.user_switch)
def user_switch(c, x, y):
    """Implement `user_switch`."""
    return x if c else y


@register(primops.scalar_to_array)
def scalar_to_array(x, t):
    """Implement `scalar_to_array`."""
    return np.array(x)


@register(primops.array_to_scalar)
def array_to_scalar(x):
    """Implement `array_to_scalar`."""
    assert isinstance(x, np.ndarray)
    return x.item()


@register(primops.broadcast_shape)
def broadcast_shape(shpx, shpy):
    """Implement `broadcast_shape`."""
    from ..abstract.data import ANYTHING
    orig_shpx = shpx
    orig_shpy = shpy
    dlen = len(shpx) - len(shpy)
    if dlen < 0:
        shpx = (1,) * -dlen + shpx
    elif dlen > 0:
        shpy = (1,) * dlen + shpy
    assert len(shpx) == len(shpy)
    shp = []
    for a, b in zip(shpx, shpy):
        if a == 1:
            shp.append(b)
        elif b == 1:
            shp.append(a)
        elif a == ANYTHING:
            shp.append(b)
        elif b == ANYTHING:
            shp.append(a)
        elif a == b:
            shp.append(a)
        else:
            raise ValueError(
                f'Cannot broadcast shapes {orig_shpx} and {orig_shpy}.'
            )
    return tuple(shp)


@register(primops.invert_permutation)
def invert_permutation(perm):
    """Implement `invert_permutation`."""
    return tuple(perm.index(i) for i in range(len(perm)))


@register(primops.make_record)
def make_record(typ, *args):
    """Implement `make_record`."""
    from ..abstract import type_to_abstract
    typ = type_to_abstract(typ)
    return typ.constructor(*args)


@register(primops.tuple_len)
@register(primops.list_len)
@register(primops.array_len)
def _len(x):
    """Implement `len`."""
    return len(x)


@register(primops.make_list)
def make_list(*xs):
    """Implement `make_list`."""
    return list(xs)


@register(primops.make_dict)
def make_dict(typ, *values):
    """Implement `make_dict`."""
    return dict(zip(typ.entries.keys(), values))


@py_register(primops.list_reduce)
def list_reduce(fn, lst, dflt):
    """Implement `list_reduce`."""
    return reduce(fn, lst, dflt)


@vm_register(primops.list_reduce)
def _list_reduce_vm(vm, fn, lst, dflt):
    def fn_(a, b):
        return vm.call(fn, [a, b])
    return list_reduce(fn_, lst, dflt)


@py_register(primops.J)
def J(x):
    """Implement `J`."""
    raise NotImplementedError()


@py_register(primops.Jinv)
def Jinv(x):
    """Implement `Jinv`."""
    raise NotImplementedError()


@register(primops.embed)
def embed(node):
    """Placeholder for the implementation of `embed`."""
    raise NotImplementedError()


@register(primops.env_setitem)
def env_setitem(env, key, x):
    """Implement `env_setitem`."""
    return env.set(key, x)


@register(primops.env_getitem)
def env_getitem(env, key, default):
    """Implement `env_getitem`."""
    return env.get(key, default)


@register(primops.env_add)
def env_add(env1, env2):
    """Implement `env_add`."""
    return env1.add(env2)


@register(primops.unsafe_static_cast)
def unsafe_static_cast(x, t):  # pragma: no cover
    """Implement `unsafe_static_cast`."""
    return x  # pragma: no cover


@register(primops.hastag)
def hastag(x, tag):
    """Implement `hastag`."""
    return x.has(tag)


@register(primops.casttag)
def casttag(x, tag):
    """Implement `casttag`."""
    return x.cast(tag)


@register(primops.tagged)
def tagged(x, tag=None):
    """Implement `tagged`."""
    if tag is None:
        return x
    else:
        return TaggedValue(tag, x)
