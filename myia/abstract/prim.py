
import numpy as np
import operator
import inspect
from functools import reduce
from collections import defaultdict
from operator import getitem

from .base import (
    ABSENT,
    AbstractBase,
    AbstractValue,
    AbstractScalar,
    AbstractType,
    AbstractFunction,
    AbstractTuple,
    AbstractArray,
    AbstractList,
    AbstractClass,
    AbstractJTagged,
    PartialApplication,
    JTransformedFunction,
    TrackableFunction,
    Possibilities,
    GraphAndContext,
    DummyFunction,
    sensitivity_transform,
    VALUE, TYPE, SHAPE,
)

from .inf import (
    AbstractTrack,
    XInferrer,
    GraphXInferrer,
)

from .. import dtype, dshape
from ..ir import Graph
from ..dtype import Number, Float, Bool
from ..infer import ANYTHING, InferenceVar, Context, MyiaTypeError, \
    InferenceError, reify, MyiaShapeError, VOID
from ..infer.core import Pending, find_coherent_result_2
from ..prim import ops as P
from ..utils import Namespace, SymbolicKeyInstance, is_dataclass_type


abstract_inferrer_constructors = {}


def reg(*prims):
    def deco(cls):
        for prim in prims:
            abstract_inferrer_constructors[prim] = cls
        return cls
    return deco


class StandardXInferrer(XInferrer):
    def __init__(self, infer):
        super().__init__()
        self._infer = infer
        data = inspect.getfullargspec(infer)
        assert data.varkw is None
        assert data.defaults is None
        assert data.kwonlyargs == []
        assert data.kwonlydefaults is None
        self.nargs = None if data.varargs else len(data.args) - 1
        self.typemap = {}
        for name, ann in data.annotations.items():
            self.typemap[data.args.index(name) - 1] = ann

    async def infer(self, track, *args):
        if self.nargs is not None and len(args) != self.nargs:
            raise MyiaTypeError('Wrong number of arguments.')
        for i, arg in enumerate(args):
            typ = self.typemap.get(i)
            if typ is None:
                pass
            elif dtype.ismyiatype(typ):
                await reify(track.chk(typ, arg.values.get(TYPE, ABSENT)))
            elif isinstance(typ, type) and issubclass(typ, AbstractBase):
                if not isinstance(arg, typ):
                    raise MyiaTypeError(
                        f'Wrong type {arg} != {typ} for {self._infer}'
                    )
            elif callable(typ):
                await reify(track.chk(typ, arg))
        return await self._infer(track, *args)


def standard_prim(*prims):
    def deco(fn):
        xinf = StandardXInferrer.partial(infer=fn)
        for prim in prims:
            abstract_inferrer_constructors[prim] = xinf
    return deco


class WithImplXInferrer(XInferrer):
    def __init__(self, impl, nolimit=False):
        super().__init__()
        self.impl = impl
        self.nolimit = nolimit
        data = inspect.getfullargspec(impl)
        assert data.varargs is None
        assert data.varkw is None
        assert data.defaults is None
        assert data.kwonlyargs == []
        assert data.kwonlydefaults is None
        self.nargs = len(data.args)
        self.impl_data = data

    def run_impl(self, track, args, outtype):
        depth = max((arg.count for arg in args), default=0)
        if not self.nolimit and depth >= track.max_depth:
            outval = ANYTHING
        else:
            values = [arg.values[VALUE] for arg in args]
            if any(v is ANYTHING for v in values):
                outval = ANYTHING
            else:
                outval = self.impl(*values)

        if outval is ANYTHING:
            depth = track.max_depth
        rval = AbstractScalar({
            VALUE: outval,
            TYPE: outtype,
            SHAPE: dshape.NOSHAPE
        })
        rval.count = min(depth + 1, track.max_depth)
        return rval


class UniformPrimitiveXInferrer(WithImplXInferrer):
    def __init__(self, impl, nolimit=False):
        super().__init__(impl, nolimit)
        data = self.impl_data
        self.typemap = defaultdict(list)
        for i, arg in enumerate(data.args):
            self.typemap[data.annotations[arg]].append(i)
        self.outtype = data.annotations['return']
        self.nolimit = nolimit

    async def infer(self, track, *args):
        if len(args) != self.nargs:
            raise MyiaTypeError('Wrong number of arguments.')

        outtype = self.outtype
        if any(not isinstance(arg, AbstractScalar) for arg in args):
            raise MyiaTypeError('Expected scalar')
        ts = [arg.values[TYPE] for arg in args]
        for typ, indexes in self.typemap.items():
            selection = [ts[i] for i in indexes]
            # res = await track.will_check(typ, *selection)
            res = track.chk(typ, *selection)
            if typ == self.outtype:
                outtype = res

        return self.run_impl(track, args, outtype)


def uniform_prim(prim, nolimit=False):
    def deco(fn):
        xinf = UniformPrimitiveXInferrer.partial(impl=fn, nolimit=nolimit)
        abstract_inferrer_constructors[prim] = xinf
    return deco


class MyiaNameError(InferenceError):
    """Raised when a name is not found in scope."""


class MyiaAttributeError(InferenceError):
    """Raised when an attribute is not found in a type or module."""


def _prim_or_graph(afn):
    from ..prim import Primitive
    from ..ir import Graph
    fns = afn.values[VALUE]
    assert isinstance(fns, Possibilities)
    assert len(fns) == 1
    fn, = fns
    if isinstance(fn, TrackableFunction):
        fn = fn.fn
    assert isinstance(fn, (Primitive, Graph))
    return fn


async def static_getter(track, data, item, fetch, on_dcattr, chk=None,
                        dataref=None, outref=None):
    """Return an inferrer for resolve or getattr.

    Arguments:
        track: The track on which the inference operates.
        data: A ref to the data.
        item: A ref to the item/attribute.
        fetch: A function to resolve the item on the data.
        chk: A function to check the values inferred for the
            data and item.
    """
    from ..infer import Reference

    resources = track.engine.pipeline.resources

    data_t = data.build(TYPE)
    item_v = item.build(VALUE, default=ANYTHING)

    if item_v is ANYTHING:
        raise InferenceError(
            'The value of the attribute could not be inferred.'
        )

    if isinstance(data_t, Pending):
        case, *args = await find_coherent_result_2(
            data_t,
            lambda t: _resolve_case(resources, t, item_v, chk)
        )
    else:
        case, *args = await _resolve_case(resources, data_t, item_v, chk)

    if case == 'class':
        # Get field from Class
        if item_v in data_t.attributes:
            return await on_dcattr(data, data_t, item_v)
        elif item_v in data_t.methods:
            method = data_t.methods[item_v]
            method = resources.convert(method)
            inferrer = track.from_value(method, Context.empty(), ref=outref)
            fn = _prim_or_graph(inferrer)
            g = outref.node.graph
            eng = outref.engine
            ref = Reference(outref.engine,
                            g.apply(P.partial, fn, dataref.node),
                            outref.context)
            return await eng.forward_reference('abstract', outref, ref)
        else:
            raise InferenceError(f'Unknown field in {data_t}: {item_v}')

    elif case == 'method':
        method, = args
        method = resources.convert(method)
        inferrer = track.from_value(method, Context.empty(), ref=outref)
        fn = _prim_or_graph(inferrer)
        g = outref.node.graph
        eng = outref.engine
        ref = Reference(outref.engine,
                        g.apply(P.partial, fn, dataref.node),
                        outref.context)
        return await eng.forward_reference('abstract', outref, ref)

    elif case == 'no_method':
        msg = f"object of type {data_t} has no attribute '{item_v}'"
        raise MyiaAttributeError(msg)

    else:
        # Module or static namespace
        data_v = data.build(VALUE, default=ANYTHING)
        if data_v is ANYTHING:
            raise InferenceError(
                'Could not infer the type or the value of the object'
                f" on which to resolve the attribute '{item_v}"
            )
        if chk:
            chk(data_v, item_v)
        try:
            raw = fetch(data_v, item_v)
        except NameError:
            raise MyiaNameError(f"Cannot resolve name '{item_v}'")
        except AttributeError as e:
            raise MyiaAttributeError(str(e))
        except Exception as e:  # pragma: no cover
            raise InferenceError(f'Unexpected error in getter: {e!r}')
        value = resources.convert(raw)
        if is_dataclass_type(value):
            typ = dtype.pytype_to_myiatype(value)
            g = outref.node.graph
            eng = outref.engine
            ref = Reference(outref.engine,
                            g.apply(P.partial, P.make_record, typ),
                            outref.context)
            return await eng.forward_reference('abstract', outref, ref)
        else:
            return track.from_value(value, Context.empty(), ref=outref)


async def _resolve_case(resources, data_t, item_v, chk):
    mmap = resources.method_map

    if dtype.ismyiatype(data_t, dtype.Class):
        return ('class', data_t)

    # Try method map
    try:
        mmap_t = mmap[data_t]
    except KeyError:
        mmap_t = None

    if mmap_t is not None:
        # Method call
        if chk:
            chk(None, item_v)
        if item_v in mmap_t:
            method = mmap_t[item_v]
            return ('method', method)
        else:
            return ('no_method',)

    return ('static',)


def _shape_type(track, shp):
    shp_t = track.chk(AbstractTuple, shp)
    for elem_t in shp_t.elements:
        # track.chk(dtype.UInt[64], elem_t.values[TYPE])
        track.abstract_merge(dtype.UInt[64], elem_t.values[TYPE])
    return shp_t


def prod(iterable):
    """Return the product of the elements of the iterator."""
    return reduce(operator.mul, iterable, 1)


async def issubtype(x, model):
    if model is dtype.Object:
        return True
    elif model is dtype.Tuple:
        return isinstance(x, AbstractTuple)
    elif model is dtype.Array:
        return isinstance(x, AbstractArray)
    elif model is dtype.List:
        return isinstance(x, AbstractList)
    elif model is dtype.Class:
        return isinstance(x, AbstractClass)

    elif dtype.ismyiatype(model, dtype.Tuple):
        return isinstance(x, AbstractTuple) \
            and len(x.elements) == len(model.elements) \
            and all([await issubtype(xe, me)
                     for xe, me in zip(x.elements, model.elements)])
    elif dtype.ismyiatype(model, dtype.Array):
        return isinstance(x, AbstractArray) \
            and await issubtype(x.element, model.elements)
    elif dtype.ismyiatype(model, dtype.List):
        return isinstance(x, AbstractList) \
            and await issubtype(x.element, model.elements)
    elif dtype.ismyiatype(model, dtype.Class):
        return isinstance(x, AbstractClass) \
            and x.tag == model.tag \
            and all([await issubtype(x.attributes[name], attr_t)
                     for name, attr_t in model.attributes.items()])

    elif dtype.ismyiatype(model, dtype.Number):
        if not isinstance(x, AbstractScalar):
            return False
        t = x.values[TYPE]
        if isinstance(t, Pending):
            async def chk(t):
                return dtype.ismyiatype(t, model)
            return await find_coherent_result_2(t, chk)
        else:
            return dtype.ismyiatype(t, model)
    else:
        return False


##############
# Arithmetic #
##############


@uniform_prim(P.scalar_add)
def _inf_scalar_add(x: Number, y: Number) -> Number:
    return x + y


@uniform_prim(P.scalar_sub)
def _inf_scalar_sub(x: Number, y: Number) -> Number:
    return x - y


@uniform_prim(P.scalar_mul)
def _inf_scalar_mul(x: Number, y: Number) -> Number:
    return x * y


@uniform_prim(P.scalar_div)
def _inf_scalar_div(x: Number, y: Number) -> Number:
    if isinstance(x, (float, np.floating)):
        return x / y
    else:
        return int(x / y)


@uniform_prim(P.scalar_mod)
def _inf_scalar_mod(x: Number, y: Number) -> Number:
    return x % y


@uniform_prim(P.scalar_pow)
def _inf_scalar_pow(x: Number, y: Number) -> Number:
    return x ** y


@uniform_prim(P.scalar_trunc)
def _inf_scalar_trunc(x: Number) -> Number:
    return np.trunc(x)


@uniform_prim(P.scalar_floor)
def _inf_scalar_floor(x: Number) -> Number:
    return np.floor(x)


@uniform_prim(P.scalar_uadd)
def _inf_scalar_uadd(x: Number) -> Number:
    return x


@uniform_prim(P.scalar_usub)
def _inf_scalar_usub(x: Number) -> Number:
    return -x


@uniform_prim(P.scalar_exp)
def _inf_scalar_exp(x: Number) -> Number:
    return math.exp(x)


@uniform_prim(P.scalar_log)
def _inf_scalar_log(x: Float) -> Float:
    return math.log(x)


@uniform_prim(P.scalar_sin)
def _inf_scalar_sin(x: Number) -> Number:
    return math.sin(x)


@uniform_prim(P.scalar_cos)
def _inf_scalar_cos(x: Number) -> Number:
    return math.cos(x)


@uniform_prim(P.scalar_tan)
def _inf_scalar_tan(x: Number) -> Number:
    return math.tan(x)


###############
# Comparisons #
###############


@uniform_prim(P.scalar_eq)
def _inf_scalar_eq(x: Number, y: Number) -> Bool:
    return x == y


@uniform_prim(P.scalar_lt)
def _inf_scalar_lt(x: Number, y: Number) -> Bool:
    return x < y


@uniform_prim(P.scalar_gt)
def _inf_scalar_gt(x: Number, y: Number) -> Bool:
    return x > y


@uniform_prim(P.scalar_ne)
def _inf_scalar_ne(x: Number, y: Number) -> Bool:
    return x != y


@uniform_prim(P.scalar_le)
def _inf_scalar_le(x: Number, y: Number) -> Bool:
    return x <= y


@uniform_prim(P.scalar_ge)
def _inf_scalar_ge(x: Number, y: Number) -> Bool:
    return x >= y


@uniform_prim(P.bool_not, nolimit=True)
def _inf_bool_not(x: Bool) -> Bool:
    return not x


@uniform_prim(P.bool_and, nolimit=True)
def _inf_bool_and(x: Bool, y: Bool) -> Bool:
    return x and y


@uniform_prim(P.bool_or, nolimit=True)
def _inf_bool_or(x: Bool, y: Bool) -> Bool:
    return x or y


@uniform_prim(P.bool_eq, nolimit=True)
def _inf_bool_eq(x: Bool, y: Bool) -> Bool:
    return x == y


######################
# Type introspection #
######################


@standard_prim(P.typeof)
async def _inf_typeof(track, value):
    t = value.build(TYPE)
    return AbstractType({
        VALUE: t,
        TYPE: dtype.TypeType,
        SHAPE: dshape.NOSHAPE
    })


@standard_prim(P.hastype)
async def _inf_hastype(track, value, model: dtype.TypeType):
    model_t = model.values[VALUE]
    if model_t is ANYTHING:
        raise MyiaTypeError('hastype must be resolvable statically')
    else:
        v = await issubtype(value, model_t)
    return AbstractScalar({
        VALUE: v,
        TYPE: dtype.Bool,
        SHAPE: dshape.NOSHAPE
    })


###################
# Data structures #
###################


@standard_prim(P.make_tuple)
async def _inf_make_tuple(track, *args):
    return AbstractTuple(args)


@standard_prim(P.make_list)
async def _inf_make_list(track, *args):
    if len(args) == 0:
        assert False
        # dflt = AbstractScalar({
        #     VALUE: ANYTHING,
        #     TYPE: dtype.Problem[VOID],
        #     SHAPE: dshape.NOSHAPE,
        # })
    else:
        res = track.abstract_merge(*args)
    return AbstractList(res)


@standard_prim(P.make_record)
async def infer_type_make_record(track, _cls: dtype.TypeType, *elems):
    """Infer the return type of make_record."""
    cls = _cls.values[VALUE]
    if cls is ANYTHING:
        raise MyiaTypeError('Expected a class to inst')
    expected = list(cls.attributes.items())
    if len(expected) != len(elems):
        raise MyiaTypeError('Wrong class inst')
    for (name, t), elem in zip(expected, elems):
        if not (await issubtype(elem, t)):
            raise MyiaTypeError('Wrong class inst')

    return AbstractClass(
        cls.tag,
        {
            name: elem
            for (name, _), elem in zip(expected, elems)
        },
        cls.methods
    )


@standard_prim(P.tuple_getitem)
async def _inf_tuple_getitem(track, arg: AbstractTuple, idx: dtype.Int[64]):
    idx_v = idx.values[VALUE]
    if idx_v is ANYTHING:
        raise MyiaTypeError(
            'Tuples must be indexed with a constant'
        )
    nelems = len(arg.elements)
    if not -nelems <= idx_v < nelems:
        raise MyiaTypeError(
            'Tuple element out of range'
        )
    return arg.elements[idx_v]


@standard_prim(P.list_getitem)
async def _inf_list_getitem(track, arg: AbstractList, idx: dtype.Int[64]):
    return arg.element


@standard_prim(P.array_getitem)
async def _inf_array_getitem(track, arg: AbstractArray, idx: dtype.Int[64]):
    return arg.element


@standard_prim(P.tuple_setitem)
async def _inf_tuple_setitem(track,
                             arg: AbstractTuple,
                             idx: dtype.Int[64],
                             value: AbstractBase):
    idx_v = idx.values[VALUE]
    if idx_v is ANYTHING:
        raise MyiaTypeError(
            'Tuples must be indexed with a constant'
        )
    nelems = len(arg.elements)
    if not -nelems <= idx_v < nelems:
        raise MyiaTypeError(
            'Tuple element out of range'
        )
    elts = arg.elements
    new_elts = tuple([*elts[:idx_v], value, *elts[idx_v + 1:]])
    return AbstractTuple(new_elts)


@standard_prim(P.list_setitem)
async def _inf_list_setitem(track,
                            arg: AbstractList,
                            idx: dtype.Int[64],
                            value: AbstractBase):
    track.abstract_merge(arg.element, value)
    return arg


@standard_prim(P.array_setitem)
async def _inf_array_setitem(track,
                             arg: AbstractArray,
                             idx: dtype.Int[64],
                             value: AbstractBase):
    track.abstract_merge(arg.element, value)
    return arg


@standard_prim(P.list_append)
async def _inf_list_append(track,
                           arg: AbstractList,
                           value: AbstractBase):
    track.abstract_merge(arg.element, value)
    return arg


class _GetAttrXInferrer(XInferrer):
    async def __call__(self, track, outref, argrefs):
        if len(argrefs) != 2:
            raise MyiaTypeError('Wrong number of arguments')
        r_data, r_item = argrefs
        data = await r_data['abstract']
        item = await r_item['abstract']

        def chk(data_v, item_v):
            if not isinstance(item_v, str):  # pragma: no cover
                raise MyiaTypeError(
                    f'item argument to resolve must be a string, not {item_v}.'
                )

        async def on_dcattr(data, data_t, item_v):
            return data.attributes[item_v]

        return await static_getter(
            track, data, item,
            fetch=getattr,
            on_dcattr=on_dcattr,
            chk=chk,
            outref=outref,
            dataref=r_data,
        )


abstract_inferrer_constructors[P.getattr] = _GetAttrXInferrer.partial()


# setattr = Primitive('setattr')


@standard_prim(P.tuple_len)
async def _inf_tuple_len(track, xs: AbstractTuple):
    return AbstractScalar({
        VALUE: len(xs.elements),
        TYPE: dtype.Int[64],
        SHAPE: dshape.NOSHAPE
    })


@standard_prim(P.list_len)
async def _inf_list_len(track, xs: AbstractList):
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.Int[64],
        SHAPE: dshape.NOSHAPE
    })


@standard_prim(P.array_len)
async def _inf_array_len(track, xs: AbstractArray):
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.Int[64],
        SHAPE: dshape.NOSHAPE
    })


@standard_prim(P.list_map)
async def _inf_list_map(track, fn, *lists):
    if len(lists) < 1:
        raise MyiaTypeError('list_map requires at least one list')
    await track.chkimm(AbstractList, *lists)
    subargs = [l.element for l in lists]
    result = await track.execute(fn, *subargs)
    return AbstractList(result)


@standard_prim(P.list_reduce)
async def _inf_list_reduce(track, fn, lst: AbstractList, dflt):
    result1 = await track.execute(fn, lst.element, lst.element)
    result2 = await track.execute(fn, dflt, lst.element)
    result = track.abstract_merge(result1, result2)
    return result


##########
# Arrays #
##########


@standard_prim(P.scalar_to_array)
async def _inf_scalar_to_array(track, a: AbstractScalar):
    return AbstractArray(a, {SHAPE: ()})


@standard_prim(P.array_to_scalar)
async def _inf_array_to_scalar(track, a: AbstractArray):
    a_shp = a.values[SHAPE]
    if len(a_shp) != 0:
        raise MyiaShapeError("array_to_scalar requires shape ()")
    return a.element


@standard_prim(P.broadcast_shape)
async def _inf_broadcast_shape(track, xs: _shape_type, ys: _shape_type):
    shp_xs_n = len(xs.elements)
    shp_ys_n = len(ys.elements)

    from ..prim.py_implementations import broadcast_shape
    shp_x = xs.build(VALUE, default=ANYTHING)
    shp_y = ys.build(VALUE, default=ANYTHING)
    elems = []
    if shp_x is ANYTHING or shp_y is ANYTHING:
        for i in range(max(shp_xs_n, shp_ys_n)):
            elems.append(AbstractScalar({
                VALUE: ANYTHING,
                TYPE: dtype.UInt[64],
                SHAPE: dshape.NOSHAPE
            }))
    else:
        try:
            res = broadcast_shape(shp_x, shp_y)
        except ValueError:
            raise MyiaTypeError('Cannot broadcast')
        for n in res:
            elems.append(AbstractScalar({
                VALUE: n,
                TYPE: dtype.UInt[64],
                SHAPE: dshape.NOSHAPE
            }))
    return AbstractTuple(elems)


@standard_prim(P.invert_permutation)
async def _inf_invert_permutation(track, perm: _shape_type):
    v = [x.values[VALUE] for x in perm.elements]
    return AbstractTuple(
        [perm.elements[v.index(i)] if i in v else AbstractScalar({
             VALUE: ANYTHING,
             TYPE: dtype.UInt[64],
             SHAPE: dshape.NOSHAPE,
         })
         for i in range(len(v))]
    )


@standard_prim(P.shape)
async def _inf_shape(track, a: AbstractArray):
    shp = await reify(a.values[SHAPE])
    values = [
        AbstractScalar({
            VALUE: entry,
            TYPE: dtype.UInt[64],
            SHAPE: dshape.NOSHAPE
        })
        for entry in shp
    ]
    return AbstractTuple(values)


@standard_prim(P.array_map)
async def _inf_array_map(track, fn: AbstractFunction, *arrays):
    if len(arrays) < 1:
        raise MyiaTypeError('array_map requires at least one array')
    await track.chkimm(AbstractArray, *arrays)
    subargs = [a.element for a in arrays]
    result = await track.execute(fn, *subargs)

    shapes = [a.values[SHAPE] for a in arrays]
    shape0, *rest = shapes
    if any(len(s) != len(shape0) for s in rest):
        raise MyiaShapeError("Expect same shapes for array_map")
    rshape = []
    for entries in zip(*shapes):
        entries = set(entries)
        entries.add(ANYTHING)
        if len(entries) == 1:
            rshape.append(ANYTHING)
        elif len(entries) == 2:
            entries.remove(ANYTHING)
            entry, = entries
            rshape.append(entry)
        else:
            raise MyiaShapeError("Expect same shapes for array_map")

    return AbstractArray(result, {SHAPE: tuple(rshape)})


# array_scan = Primitive('array_scan')


@standard_prim(P.array_reduce)
async def _inf_array_reduce(track,
                            fn: AbstractFunction,
                            a: AbstractArray,
                            shp: _shape_type):

    shp_i = await reify(a.values[SHAPE])
    shp_v = shp.build(VALUE, default=ANYTHING)
    if shp_v == ANYTHING:
        raise AssertionError(
            'We currently require knowing the shape for reduce.'
        )
        # return (ANYTHING,) * (len(shp_i) - 1)
    else:
        delta = len(shp_i) - len(shp_v)
        if delta < 0 \
                or any(1 != s1 != ANYTHING and 1 != s2 != ANYTHING and s1 != s2
                       for s1, s2 in zip(shp_i[delta:], shp_v)):
            raise MyiaShapeError(
                f'Incompatible dims for reduce: {shp_i}, {shp_v}'
            )

    res = await track.execute(fn, a.element, a.element)
    return AbstractArray(res, {SHAPE: shp_v})


@standard_prim(P.distribute)
async def _inf_distribute(track, a: AbstractArray, _shp: _shape_type):
    shp = _shp.build(VALUE, default=ANYTHING)
    if shp == ANYTHING:
        shp = (ANYTHING,) * len(_shp.elements)
    a_shp = await reify(a.values[SHAPE])
    delta = len(shp) - len(a_shp)
    if delta < 0:
        raise MyiaShapeError("Cannot distribute to smaller shape")
    elif delta > 0:
        a_shp = (1,) * delta + a_shp
    for vs, s in zip(a_shp, shp):
        if vs != s and vs not in (1, ANYTHING) and s not in (1, ANYTHING):
            raise MyiaShapeError("Cannot change shape when distributing")
    return AbstractArray(a.element, {SHAPE: shp})


@standard_prim(P.reshape)
async def _inf_reshape(track, a: AbstractArray, _shp: _shape_type):
    shp = _shp.build(VALUE, default=ANYTHING)
    if shp == ANYTHING:
        shp = (ANYTHING,) * len(_shp.elements)
    a_shp = await reify(a.values[SHAPE])
    if (all(s is not ANYTHING for s in shp) and
        all(s is not ANYTHING for s in a_shp) and
            prod(shp) != prod(a_shp)):
        raise MyiaShapeError("Cannot change the total number of elements "
                             "in reshape")
    return AbstractArray(a.element, {SHAPE: shp})


@standard_prim(P.transpose)
async def _inf_transpose(track, a: AbstractArray, permutation: _shape_type):
    perm = permutation.build(VALUE, default=ANYTHING)
    if perm == ANYTHING:
        shp = (ANYTHING,) * len(permutation.elements)
    else:
        a_shp = await reify(a.values[SHAPE])
        if list(sorted(perm)) != list(range(len(a_shp))):
            raise MyiaShapeError(
                'The second argument of transpose must be a permutation of'
                ' all of the array\'s axes.',
                refs=[permutation]
            )

        shp = tuple(a_shp[i] for i in perm)
    return AbstractArray(a.element, {SHAPE: shp})


@standard_prim(P.dot)
async def _inf_dot(track, a: AbstractArray, b: AbstractArray):
    a_shp = a.values[SHAPE]
    b_shp = b.values[SHAPE]
    if len(a_shp) != 2 or len(b_shp) != 2:
        raise MyiaShapeError("dot needs matrix inputs")
    if (a_shp[1] != b_shp[0] and
            a_shp[1] is not ANYTHING and b_shp[0] is not ANYTHING):
        raise MyiaShapeError(
            f"Incompatible shapes in dot: {a_shp} and {b_shp}"
        )
    track.abstract_merge(a.element, b.element)
    c_shp = (a_shp[0], b_shp[1])
    return AbstractArray(a.element, {SHAPE: c_shp})


##############
# Statements #
##############


@standard_prim(P.switch)
async def _inf_switch(track, cond: Bool, tb, fb):
    v = cond.values[VALUE]
    if v is True:
        return tb
    elif v is False:
        return fb
    elif v is ANYTHING:
        return track.abstract_merge(tb, fb)
    else:
        raise AssertionError(f"Invalid condition value for switch: {v}")


#################
# Miscellaneous #
#################


@standard_prim(P.scalar_cast)
async def _inf_scalar_cast(track,
                           scalar: Number,
                           typ: AbstractType):
    t = typ.values[VALUE]
    if t is ANYTHING:
        raise MyiaTypeError('Must have concrete type for scalar_cast')
    track.chk(Number, t)
    values = {**scalar.values, TYPE: t}
    return AbstractScalar(values)


@standard_prim(P.identity, P.return_)
async def _inf_identity(track, x):
    return x


class _ResolveXInferrer(XInferrer):
    async def __call__(self, track, outref, argrefs):
        if len(argrefs) != 2:
            raise MyiaTypeError('Wrong number of arguments')
        r_data, r_item = argrefs
        data = await r_data['abstract']
        item = await r_item['abstract']

        def chk(data_v, item_v):
            if not isinstance(data_v, Namespace):  # pragma: no cover
                raise MyiaTypeError(
                    f'data argument to resolve must be Namespace, not {data_v}',
                    refs=[data]
                )
            if not isinstance(item_v, str):  # pragma: no cover
                raise MyiaTypeError(
                    f'item argument to resolve must be a string, not {item_v}.',
                    refs=[item]
                )

        async def on_dcattr(data, data_t, item_v):  # pragma: no cover
            raise MyiaTypeError('Cannot resolve on Class.')

        return await static_getter(
            track, data, item,
            fetch=getitem,
            on_dcattr=on_dcattr,
            chk=chk,
            outref=outref,
            dataref=r_data,
        )


abstract_inferrer_constructors[P.resolve] = _ResolveXInferrer.partial()


@standard_prim(P.partial)
async def _inf_partial(track, fn, *args):
    fns = fn.values[VALUE]
    assert isinstance(fns, Possibilities)
    return AbstractFunction(*[
        PartialApplication(fn, args) for fn in fns
    ])


class _EmbedXInferrer(XInferrer):
    async def __call__(self, track, outref, argrefs):
        if len(argrefs) != 1:
            raise MyiaTypeError('Wrong number of arguments')
        xref, = argrefs
        x = await xref['abstract']
        key = SymbolicKeyInstance(xref.node, sensitivity_transform(x))
        return AbstractScalar({
            VALUE: key,
            TYPE: dtype.SymbolicKeyType,
            SHAPE: dshape.NOSHAPE,
        })


abstract_inferrer_constructors[P.embed] = _EmbedXInferrer.partial()


@standard_prim(P.env_getitem)
async def _inf_env_getitem(track,
                           env: dtype.EnvType,
                           key: dtype.SymbolicKeyType,
                           dflt):
    expected = key.values[VALUE].inferred
    track.abstract_merge(expected, dflt)
    return expected


@standard_prim(P.env_setitem)
async def _inf_env_setitem(track,
                           env: dtype.EnvType,
                           key: dtype.SymbolicKeyType,
                           value):
    expected = key.values[VALUE].inferred
    track.abstract_merge(expected, value)
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.EnvType,
        SHAPE: dshape.NOSHAPE,
    })


@standard_prim(P.env_add)
async def _inf_env_add(track, env1, env2):
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.EnvType,
        SHAPE: dshape.NOSHAPE,
    })


@standard_prim(P.J)
async def _inf_J(track, x):
    if isinstance(x, AbstractFunction):
        v = await x.get()
        return AbstractFunction(*[JTransformedFunction(poss)
                                  for poss in v])
    return AbstractJTagged(x)


@standard_prim(P.Jinv)
async def _inf_Jinv(track, x):
    if isinstance(x, AbstractFunction):
        v = await x.get()
        results = []
        for f in v:
            if isinstance(f, TrackableFunction):
                f = f.fn
            if isinstance(f, JTransformedFunction):
                results.append(f.fn)
            elif isinstance(f, (Graph, GraphAndContext)):
                if isinstance(f, GraphAndContext):
                    g = f.graph
                else:
                    g = f
                primal = g and g.transforms.get('primal', None)
                if primal:
                    primal = track.engine.pipeline.resources.convert(primal)
                    if isinstance(primal, Graph) and primal.parent:
                        # The primal for a closure can't be used because it points to
                        # the original nodes of its parent, whereas we would like to
                        # point to the transformed nodes of the parent. This is
                        # fixable, and will need to be fixed to support a few edge
                        # cases.
                        results.append(DummyFunction())
                    else:
                        results.append(primal)
                else:
                    raise MyiaTypeError(f'Bad input type for Jinv: {f}')
            else:
                raise MyiaTypeError(
                    f'Expected JTransformedFunction, not {f}'
                )
        return AbstractFunction(*results)
    if isinstance(x, AbstractJTagged):
        return x.element
    else:
        raise MyiaTypeError('Expected JTagged')
