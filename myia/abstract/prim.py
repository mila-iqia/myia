"""Inferrers for primitives."""

import operator
import inspect
from functools import reduce
from collections import defaultdict
from operator import getitem

from .. import dtype
from ..ir import Graph
from ..dtype import Number, Bool
from ..prim import ops as P, Primitive, py_implementations as py
from ..utils import Namespace, SymbolicKeyInstance, is_dataclass_type


from .data import (
    ANYTHING,
    AbstractValue,
    AbstractScalar,
    AbstractType,
    AbstractFunction,
    AbstractTuple,
    AbstractArray,
    AbstractList,
    AbstractClass,
    AbstractJTagged,
    AbstractUnion,
    abstract_union,
    PartialApplication,
    JTransformedFunction,
    PrimitiveFunction,
    Possibilities,
    GraphFunction,
    DummyFunction,
    VALUE, TYPE, SHAPE,
    MyiaTypeError, InferenceError, MyiaShapeError, check_nargs,
    infer_trace
)
from .loop import Pending, find_coherent_result, force_pending
from .ref import Context, ConditionalContext, Reference
from .utils import sensitivity_transform, build_value, build_type, broaden
from .infer import Inferrer, to_abstract


abstract_inferrer_constructors = {}


class StandardInferrer(Inferrer):
    """Generic inferrer for primitives.

    Arguments:
        infer: The inference function. Its arguments and type annotations
            will be inspected and checked automatically.
    """

    def __init__(self, prim, infer):
        """Initialize a StandardInferrer."""
        super().__init__()
        self.prim = prim
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

    async def infer(self, engine, *args):
        """Infer the abstract result given the abstract arguments."""
        check_nargs(self.prim, self.nargs, args)
        infer_trace.set({**infer_trace.get(), self.prim: (self.prim, args)})
        for i, arg in enumerate(args):
            typ = self.typemap.get(i)
            if typ is None:
                pass
            elif dtype.ismyiatype(typ):
                await force_pending(engine.check(typ, build_type(arg)))
            elif isinstance(typ, type) and issubclass(typ, AbstractValue):
                if not isinstance(arg, typ):
                    raise MyiaTypeError(
                        f'Wrong type {arg} != {typ} for {self._infer}'
                    )
            elif callable(typ):
                await force_pending(engine.check(typ, arg))
        return await self._infer(engine, *args)


def standard_prim(prim):
    """Decorator to define and register a StandardInferrer."""
    def deco(fn):
        xinf = StandardInferrer.partial(prim=prim, infer=fn)
        abstract_inferrer_constructors[prim] = xinf
    return deco


class WithImplInferrer(Inferrer):
    """Inferrer derived from an implementation.

    Arguments:
        impl: The implementation.
        infer_value: Whether to do constant propagation through this
            implementation.
    """

    def __init__(self, prim, impl, infer_value=False):
        """Initialize a WithImplInferrer."""
        super().__init__()
        self.prim = prim
        self.impl = impl
        self.infer_value = infer_value
        data = inspect.getfullargspec(impl)
        assert data.varargs is None
        assert data.varkw is None
        assert data.defaults is None
        assert data.kwonlyargs == []
        assert data.kwonlydefaults is None
        self.nargs = len(data.args)
        self.impl_data = data

    def run_impl(self, engine, args, outtype):
        """Run the implementation on abstract data.

        If infer_value is False, this returns an AbstractScalar with value
        ANYTHING.

        Arguments: engine: The InferenceEngine args: The abstract arguments
            outtype: The output type to give to the result
        """
        if not self.infer_value:
            outval = ANYTHING
        else:
            values = [arg.values[VALUE] for arg in args]
            if any(v is ANYTHING for v in values):
                outval = ANYTHING
            else:
                outval = self.impl(*values)

        return AbstractScalar({
            VALUE: outval,
            TYPE: outtype,
        })


class UniformPrimitiveInferrer(WithImplInferrer):
    """Inferrer derived from an implementation, requiring uniform types.

    If multiple arguments are AbstractScalars, they will all be required
    to have the same type, e.g. all Int[64] or all Float[32].

    Arguments:
        impl: The implementation.
        infer_value: Whether to do constant propagation through this
            implementation.
    """

    def __init__(self, prim, impl, infer_value=False):
        """Initialize a UniformPrimitiveInferrer."""
        super().__init__(prim, impl, infer_value)
        data = self.impl_data
        self.typemap = defaultdict(list)
        # We group the arguments with the same types together.
        for i, arg in enumerate(data.args):
            self.typemap[data.annotations[arg]].append(i)
        self.outtype = data.annotations['return']
        self.infer_value = infer_value

    def normalize_args(self, args):
        """If infer_value is False, return broadened arguments."""
        if not self.infer_value:
            args = tuple(broaden(a, None) for a in args)
        return args

    async def infer(self, engine, *args):
        """Infer the abstract result given the abstract arguments."""
        check_nargs(self.prim, self.nargs, args)
        infer_trace.set({**infer_trace.get(), self.prim: (self.prim, args)})
        outtype = self.outtype
        if any(not isinstance(arg, AbstractScalar) for arg in args):
            raise MyiaTypeError('Expected scalar')
        ts = [arg.values[TYPE] for arg in args]
        for typ, indexes in self.typemap.items():
            # Each group is merged using check
            selection = [ts[i] for i in indexes]
            res = engine.check(typ, *selection)
            if typ == self.outtype:
                outtype = res

        return self.run_impl(engine, args, outtype)


def uniform_prim(prim, infer_value=False):
    """Decorator to define and register a UniformPrimitiveInferrer.

    Arguments:
        prim: The primitive for which the inferrer is defined.
        infer_value: Whether to limit constant propagation through this
            operation or not.
    """
    def deco(fn):
        xinf = UniformPrimitiveInferrer.partial(
            prim=prim,
            impl=fn,
            infer_value=infer_value
        )
        abstract_inferrer_constructors[prim] = xinf
    return deco


class MyiaNameError(InferenceError):
    """Raised when a name is not found in scope."""


class MyiaAttributeError(InferenceError):
    """Raised when an attribute is not found in a type or module."""


def _prim_or_graph(afn):
    # Check that afn represents a single Primitive/Graph and return it.
    assert isinstance(afn, AbstractFunction)
    fn = afn.get_unique()
    if isinstance(fn, PrimitiveFunction):
        fn = fn.prim
    if isinstance(fn, GraphFunction):
        assert fn.context == Context.empty()
        fn = fn.graph
    assert isinstance(fn, (Primitive, Graph))
    return fn


async def static_getter(engine, data, item, fetch, on_dcattr, chk=None,
                        dataref=None, outref=None):
    """Return an inferrer for resolve or getattr.

    Arguments:
        engine: The engine on which the inference operates.
        data: A ref to the data.
        item: A ref to the item/attribute.
        fetch: A function to resolve the item on the data.
        on_dcattr: A function to resolve an attribute on a dataclass.
        chk: A function to check the values inferred for the
            data and item.
        dataref: The Reference to the data.
        outref: The Reference to the output.
    """
    resources = engine.pipeline.resources

    data_t = build_type(data)
    item_v = build_value(item, default=ANYTHING)

    if item_v is ANYTHING:
        raise InferenceError(
            'The value of the attribute could not be inferred.'
        )

    if isinstance(data_t, Pending):
        case, *args = await find_coherent_result(
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
            inferrer = to_abstract(method, Context.empty(), ref=outref)
            fn = _prim_or_graph(inferrer)
            g = outref.node.graph
            eng = outref.engine
            ref = eng.ref(g.apply(P.partial, fn, dataref.node),
                          outref.context)
            return await eng.reroute(outref, ref)
        else:
            raise InferenceError(f'Unknown field in {data_t}: {item_v}')

    elif case == 'method':
        method, = args
        method = resources.convert(method)
        inferrer = to_abstract(method, Context.empty(), ref=outref)
        fn = _prim_or_graph(inferrer)
        g = outref.node.graph
        eng = outref.engine
        ref = eng.ref(g.apply(P.partial, fn, dataref.node),
                      outref.context)
        return await eng.reroute(outref, ref)

    elif case == 'no_method':
        msg = f"object of type {data_t} has no attribute '{item_v}'"
        raise MyiaAttributeError(msg)

    else:
        # Module or static namespace
        data_v = build_value(data, default=ANYTHING)
        if data_v is ANYTHING:
            raise InferenceError(
                'Could not infer the type or the value of the object'
                f" on which to resolve the attribute '{item_v}'"
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
            ref = eng.ref(g.apply(P.partial, P.make_record, typ),
                          outref.context)
            return await eng.reroute(outref, ref)
        else:
            return to_abstract(value, Context.empty(), ref=outref)


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


def _shape_type(engine, shp):
    shp_t = engine.check(AbstractTuple, shp)
    for elem_t in shp_t.elements:
        engine.abstract_merge(dtype.UInt[64], elem_t.values[TYPE])
    return shp_t


def prod(iterable):
    """Return the product of the elements of the iterator."""
    return reduce(operator.mul, iterable, 1)


async def issubtype(x, model):
    """Check whether type x is a subtype of model."""
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
            and await issubtype(x.element, model.element_type)
    elif dtype.ismyiatype(model, dtype.Class):
        return isinstance(x, AbstractClass) \
            and x.tag == model.tag \
            and all([await issubtype(x.attributes[name], attr_t)
                     for name, attr_t in model.attributes.items()])

    elif dtype.ismyiatype(model, dtype.Number) \
            or dtype.ismyiatype(model, dtype.Bool) \
            or dtype.ismyiatype(model, dtype.Nil):
        if not isinstance(x, AbstractScalar):
            return False
        t = x.values[TYPE]
        if isinstance(t, Pending):
            async def chk(t):
                return dtype.ismyiatype(t, model)
            return await find_coherent_result(t, chk)
        else:
            return dtype.ismyiatype(t, model)
    else:
        raise AssertionError(f'Invalid model: {model}')


##############
# Arithmetic #
##############


uniform_prim(P.scalar_add)(py.scalar_add)
uniform_prim(P.scalar_sub)(py.scalar_sub)
uniform_prim(P.scalar_mul)(py.scalar_mul)
uniform_prim(P.scalar_div)(py.scalar_div)
uniform_prim(P.scalar_mod)(py.scalar_mod)
uniform_prim(P.scalar_pow)(py.scalar_pow)
uniform_prim(P.scalar_trunc)(py.scalar_trunc)
uniform_prim(P.scalar_floor)(py.scalar_floor)
uniform_prim(P.scalar_uadd, infer_value=True)(py.scalar_uadd)
uniform_prim(P.scalar_usub, infer_value=True)(py.scalar_usub)
uniform_prim(P.scalar_exp)(py.scalar_exp)
uniform_prim(P.scalar_log)(py.scalar_log)
uniform_prim(P.scalar_sin)(py.scalar_sin)
uniform_prim(P.scalar_cos)(py.scalar_cos)
uniform_prim(P.scalar_tan)(py.scalar_tan)
uniform_prim(P.scalar_tanh)(py.scalar_tanh)


###############
# Comparisons #
###############


uniform_prim(P.scalar_eq, infer_value=True)(py.scalar_eq)
uniform_prim(P.scalar_lt, infer_value=True)(py.scalar_lt)
uniform_prim(P.scalar_gt, infer_value=True)(py.scalar_gt)
uniform_prim(P.scalar_ne, infer_value=True)(py.scalar_ne)
uniform_prim(P.scalar_le, infer_value=True)(py.scalar_le)
uniform_prim(P.scalar_ge, infer_value=True)(py.scalar_ge)
uniform_prim(P.bool_not, infer_value=True)(py.bool_not)
uniform_prim(P.bool_and, infer_value=True)(py.bool_and)
uniform_prim(P.bool_or, infer_value=True)(py.bool_or)
uniform_prim(P.bool_eq, infer_value=True)(py.bool_eq)


######################
# Type introspection #
######################


async def _split_type(t, model):
    """Checks t against the model and return matching/non-matching subtypes.

    * If t is a Union, return a Union that fully matches model, and a Union
      that does not match model. No matches in either case returns None for
      that case.
    * Otherwise, return (t, None) or (None, t) depending on whether t matches
      the model.
    """
    if isinstance(t, AbstractUnion):
        matching = [(opt, await issubtype(opt, model))
                    for opt in t.options]
        t1 = abstract_union([opt for opt, m in matching if m])
        t2 = abstract_union([opt for opt, m in matching if not m])
        return t1, t2
    elif (await issubtype(t, model)):
        return t, None
    else:
        return None, t


@standard_prim(P.typeof)
async def _inf_typeof(engine, value):
    t = build_type(value)
    return AbstractType(t)


@standard_prim(P.hastype)
async def _inf_hastype(engine, value, model: dtype.TypeType):
    model_t = model.values[VALUE]
    if model_t is ANYTHING:
        raise MyiaTypeError('hastype must be resolvable statically')
    match, nomatch = await _split_type(value, model_t)
    if match is None:
        v = False
    elif nomatch is None:
        v = True
    else:
        v = ANYTHING
    return AbstractScalar({
        VALUE: v,
        TYPE: dtype.Bool,
    })


###################
# Data structures #
###################


@standard_prim(P.make_tuple)
async def _inf_make_tuple(engine, *args):
    return AbstractTuple(args)


@standard_prim(P.make_list)
async def _inf_make_list(engine, *args):
    if len(args) == 0:
        raise NotImplementedError('Cannot make empty lists.')
    else:
        res = engine.abstract_merge(*args)
    return AbstractList(res)


@standard_prim(P.make_record)
async def infer_type_make_record(engine, _cls: dtype.TypeType, *elems):
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
async def _inf_tuple_getitem(engine, arg: AbstractTuple, idx: dtype.Int[64]):
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
async def _inf_list_getitem(engine, arg: AbstractList, idx: dtype.Int[64]):
    return arg.element


@standard_prim(P.tuple_setitem)
async def _inf_tuple_setitem(engine,
                             arg: AbstractTuple,
                             idx: dtype.Int[64],
                             value: AbstractValue):
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
async def _inf_list_setitem(engine,
                            arg: AbstractList,
                            idx: dtype.Int[64],
                            value: AbstractValue):
    engine.abstract_merge(arg.element, value)
    return arg


@standard_prim(P.list_append)
async def _inf_list_append(engine,
                           arg: AbstractList,
                           value: AbstractValue):
    engine.abstract_merge(arg.element, value)
    return arg


class _GetAttrInferrer(Inferrer):
    async def run(self, engine, outref, argrefs):
        check_nargs(P.getattr, 2, argrefs)
        r_data, r_item = argrefs
        data = await r_data.get()
        item = await r_item.get()

        def chk(data_v, item_v):
            if not isinstance(item_v, str):  # pragma: no cover
                raise MyiaTypeError(
                    f'item argument to resolve must be a string, not {item_v}.'
                )

        async def on_dcattr(data, data_t, item_v):
            return data.attributes[item_v]

        rval = await static_getter(
            engine, data, item,
            fetch=getattr,
            on_dcattr=on_dcattr,
            chk=chk,
            outref=outref,
            dataref=r_data,
        )
        self.cache[(data, item)] = rval
        return rval


abstract_inferrer_constructors[P.getattr] = _GetAttrInferrer.partial()


# TODO: setattr


@standard_prim(P.tuple_len)
async def _inf_tuple_len(engine, xs: AbstractTuple):
    return AbstractScalar({
        VALUE: len(xs.elements),
        TYPE: dtype.Int[64],
    })


@standard_prim(P.list_len)
async def _inf_list_len(engine, xs: AbstractList):
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.Int[64],
    })


@standard_prim(P.array_len)
async def _inf_array_len(engine, xs: AbstractArray):
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.Int[64],
    })


@standard_prim(P.list_map)
async def _inf_list_map(engine, fn, *lists):
    if len(lists) < 1:  # pragma: no cover
        raise MyiaTypeError('list_map requires at least one list')
    await engine.check_immediate(AbstractList, *lists)
    subargs = [l.element for l in lists]
    result = await engine.execute(fn, *subargs)
    return AbstractList(result)


@standard_prim(P.list_reduce)
async def _inf_list_reduce(engine, fn, lst: AbstractList, dflt):
    result1 = await engine.execute(fn, lst.element, lst.element)
    result2 = await engine.execute(fn, dflt, lst.element)
    result = engine.abstract_merge(result1, result2)
    return result


##########
# Arrays #
##########


@standard_prim(P.scalar_to_array)
async def _inf_scalar_to_array(engine, a: AbstractScalar):
    return AbstractArray(a, {SHAPE: ()})


@standard_prim(P.array_to_scalar)
async def _inf_array_to_scalar(engine, a: AbstractArray):
    a_shp = a.values[SHAPE]
    if len(a_shp) != 0:
        raise MyiaShapeError("array_to_scalar requires shape ()")
    return a.element


@standard_prim(P.broadcast_shape)
async def _inf_broadcast_shape(engine, xs: _shape_type, ys: _shape_type):
    from ..prim.py_implementations import broadcast_shape
    shp_x = tuple(x.values[VALUE] for x in xs.elements)
    shp_y = tuple(y.values[VALUE] for y in ys.elements)
    elems = []
    try:
        res = broadcast_shape(shp_x, shp_y)
    except ValueError as e:
        raise MyiaShapeError(e.args[0])
    for n in res:
        elems.append(AbstractScalar({
            VALUE: n,
            TYPE: dtype.UInt[64],
        }))
    return AbstractTuple(elems)


@standard_prim(P.invert_permutation)
async def _inf_invert_permutation(engine, perm: _shape_type):
    v = [x.values[VALUE] for x in perm.elements]
    return AbstractTuple(
        [perm.elements[i] if i in v else AbstractScalar({
             VALUE: ANYTHING,
             TYPE: dtype.UInt[64],
         })
         for i in range(len(v))]
    )


@standard_prim(P.shape)
async def _inf_shape(engine, a: AbstractArray):
    shp = await force_pending(a.values[SHAPE])
    values = [
        AbstractScalar({
            VALUE: entry,
            TYPE: dtype.UInt[64],
        })
        for entry in shp
    ]
    return AbstractTuple(values)


@standard_prim(P.array_map)
async def _inf_array_map(engine, fn: AbstractFunction, *arrays):
    if len(arrays) < 1:
        raise MyiaTypeError('array_map requires at least one array')
    for arr in arrays:
        await engine.check_immediate(AbstractArray, arr)

    subargs = [a.element for a in arrays]
    result = await engine.execute(fn, *subargs)

    shapes = [a.values[SHAPE] for a in arrays]
    shape0, *rest = shapes
    if any(len(s) != len(shape0) for s in rest):  # pragma: no cover
        # check_immediate above is checking this for us, although
        # the error message is poor
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


# TODO: array_scan


@standard_prim(P.array_reduce)
async def _inf_array_reduce(engine,
                            fn: AbstractFunction,
                            a: AbstractArray,
                            shp: _shape_type):

    shp_i = await force_pending(a.values[SHAPE])
    shp_v = build_value(shp, default=ANYTHING)
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

    res = await engine.execute(fn, a.element, a.element)
    return AbstractArray(res, {SHAPE: shp_v})


@standard_prim(P.distribute)
async def _inf_distribute(engine, a: AbstractArray, _shp: _shape_type):
    shp = tuple(x.values[VALUE] for x in _shp.elements)
    a_shp = await force_pending(a.values[SHAPE])
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
async def _inf_reshape(engine, a: AbstractArray, _shp: _shape_type):
    shp = build_value(_shp, default=ANYTHING)
    if shp == ANYTHING:
        shp = (ANYTHING,) * len(_shp.elements)
    a_shp = await force_pending(a.values[SHAPE])
    if (all(s is not ANYTHING for s in shp) and
        all(s is not ANYTHING for s in a_shp) and
            prod(shp) != prod(a_shp)):
        raise MyiaShapeError("Cannot change the total number of elements "
                             "in reshape")
    return AbstractArray(a.element, {SHAPE: shp})


@standard_prim(P.transpose)
async def _inf_transpose(engine, a: AbstractArray, permutation: _shape_type):
    perm = build_value(permutation, default=ANYTHING)
    if perm == ANYTHING:
        shp = (ANYTHING,) * len(permutation.elements)
    else:
        a_shp = await force_pending(a.values[SHAPE])
        if list(sorted(perm)) != list(range(len(a_shp))):
            raise MyiaShapeError(
                'The second argument of transpose must be a permutation of'
                ' all of the array\'s axes.',
                refs=[permutation]
            )

        shp = tuple(a_shp[i] for i in perm)
    return AbstractArray(a.element, {SHAPE: shp})


@standard_prim(P.dot)
async def _inf_dot(engine, a: AbstractArray, b: AbstractArray):
    a_shp = a.values[SHAPE]
    b_shp = b.values[SHAPE]
    if len(a_shp) != 2 or len(b_shp) != 2:
        raise MyiaShapeError("dot needs matrix inputs")
    if (a_shp[1] != b_shp[0] and
            a_shp[1] is not ANYTHING and b_shp[0] is not ANYTHING):
        raise MyiaShapeError(
            f"Incompatible shapes in dot: {a_shp} and {b_shp}"
        )
    engine.abstract_merge(a.element, b.element)
    c_shp = (a_shp[0], b_shp[1])
    return AbstractArray(a.element, {SHAPE: c_shp})


##############
# Statements #
##############


class _SwitchInferrer(Inferrer):

    async def _special_hastype(self, engine, outref,
                               xref, typref, tbref, fbref):
        """Handle `switch(hastype(x, typ), tb, fb)`.

        We want to evaluate tb in a context where x has type typ and fb
        in a context where it doesn't.
        """
        async def wrap(branch_ref, branch_type):
            node = branch_ref.node
            if not node.is_constant_graph():
                raise MyiaTypeError(
                    'Branches of switch must be functions when the condition'
                    ' is hastype on a Union.'
                )
            branch_ctx = ConditionalContext(ctx, (xref.node, branch_type))
            new_branch_ref = engine.ref(node, branch_ctx)
            new_x = g.apply(P.unsafe_static_cast, xref.node, branch_type)
            await engine.reroute(
                engine.ref(xref.node, branch_ctx),
                engine.ref(new_x, ctx)
            )
            if branch_ref != new_branch_ref:
                fn = await engine.reroute(branch_ref, new_branch_ref)
            else:
                fn = await branch_ref.get()
            return fn.get_unique()

        ctx = outref.context
        g = xref.node.graph

        fulltype = await xref.get()
        typ = (await typref.get()).values[VALUE]
        tbtyp, fbtyp = await _split_type(fulltype, typ)
        # We are not supposed to be here if only one branch could be taken.
        assert tbtyp is not None
        assert fbtyp is not None
        return AbstractFunction(
            await wrap(tbref, tbtyp),
            await wrap(fbref, fbtyp)
        )

    async def _find_op(self, engine, condref):
        """Find a primitive operator to use for the condition.

        This skips over the bool primitive or composite used for this
        pipeline.
        """
        ctx = condref.context
        if condref.node.is_apply():
            opnode, *args = condref.node.inputs
            opref = engine.ref(opnode, ctx)
            try:
                op = (await opref.get()).get_unique()
            except MyiaTypeError:
                return None, None
            if isinstance(op, PrimitiveFunction):
                op = op.prim
            elif isinstance(op, GraphFunction):
                op = op.graph

            converter = engine.pipeline.resources.convert
            converted_bool = converter.object_map.get(bool, None)
            if op is converted_bool:
                return await self._find_op(engine, engine.ref(args[0], ctx))
            elif isinstance(op, Primitive):
                return op, [engine.ref(a, ctx) for a in args]
            else:
                return None, None
        else:
            return None, None

    async def run(self, engine, outref, argrefs):
        res = await self._run_helper(engine, outref, argrefs)
        # args = tuple([await ref.get() for ref in argrefs])
        # self.cache[args] = res
        return res

    async def _run_helper(self, engine, outref, argrefs):
        check_nargs(P.switch, 3, argrefs)
        condref, tbref, fbref = argrefs

        cond = await condref.get()
        await force_pending(engine.check(Bool, build_type(cond)))

        v = cond.values[VALUE]
        if v is True:
            return await tbref.get()
        elif v is False:
            return await fbref.get()
        elif v is ANYTHING:
            op, args = await self._find_op(engine, condref)
            method = op and getattr(self, f'_special_{op}', None)
            if method is None:
                tb = await tbref.get()
                fb = await fbref.get()
                return engine.abstract_merge(tb, fb)
            else:
                return await method(engine, outref, *args, tbref, fbref)
        else:
            raise AssertionError(f"Invalid condition value for switch: {v}")


abstract_inferrer_constructors[P.switch] = _SwitchInferrer.partial()


#################
# Miscellaneous #
#################


@standard_prim(P.scalar_cast)
async def _inf_scalar_cast(engine,
                           scalar: Number,
                           typ: AbstractType):
    t = typ.values[VALUE]
    if t is ANYTHING:
        raise MyiaTypeError('Must have concrete type for scalar_cast')
    engine.check(Number, t)
    values = {**scalar.values, TYPE: t}
    return AbstractScalar(values)


@standard_prim(P.identity)
async def _inf_identity(engine, x):
    return x


@standard_prim(P.return_)
async def _inf_return(engine, x):
    return x


class _ResolveInferrer(Inferrer):
    async def run(self, engine, outref, argrefs):
        check_nargs(P.resolve, 2, argrefs)
        r_data, r_item = argrefs
        data = await r_data.get()
        item = await r_item.get()

        def chk(data_v, item_v):
            if not isinstance(data_v, Namespace):  # pragma: no cover
                raise MyiaTypeError(
                    f'data argument to resolve must be Namespace,'
                    f' not {data_v}',
                    refs=[data]
                )
            if not isinstance(item_v, str):  # pragma: no cover
                raise MyiaTypeError(
                    f'item argument to resolve must be a string,'
                    f' not {item_v}.',
                    refs=[item]
                )

        async def on_dcattr(data, data_t, item_v):  # pragma: no cover
            raise MyiaTypeError('Cannot resolve on Class.')

        rval = await static_getter(
            engine, data, item,
            fetch=getitem,
            on_dcattr=on_dcattr,
            chk=chk,
            outref=outref,
            dataref=r_data,
        )
        self.cache[(data, item)] = rval
        return rval


abstract_inferrer_constructors[P.resolve] = _ResolveInferrer.partial()


@standard_prim(P.partial)
async def _inf_partial(engine, fn, *args):
    fns = await fn.get()
    assert isinstance(fns, Possibilities)
    return AbstractFunction(*[
        PartialApplication(fn, args) for fn in fns
    ])


class _EmbedInferrer(Inferrer):
    async def run(self, engine, outref, argrefs):
        check_nargs(P.embed, 1, argrefs)
        xref, = argrefs
        x = await xref.get()
        key = SymbolicKeyInstance(xref.node, sensitivity_transform(x))
        return AbstractScalar({
            VALUE: key,
            TYPE: dtype.SymbolicKeyType,
        })


abstract_inferrer_constructors[P.embed] = _EmbedInferrer.partial()


@standard_prim(P.env_getitem)
async def _inf_env_getitem(engine,
                           env: dtype.EnvType,
                           key: dtype.SymbolicKeyType,
                           dflt):
    expected = key.values[VALUE].abstract
    engine.abstract_merge(expected, dflt)
    return expected


@standard_prim(P.env_setitem)
async def _inf_env_setitem(engine,
                           env: dtype.EnvType,
                           key: dtype.SymbolicKeyType,
                           value):
    expected = key.values[VALUE].abstract
    engine.abstract_merge(expected, value)
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.EnvType,
    })


@standard_prim(P.env_add)
async def _inf_env_add(engine, env1, env2):
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.EnvType,
    })


@standard_prim(P.unsafe_static_cast)
async def _inf_unsafe_static_cast(engine, x, typ: AbstractScalar):
    v = typ.values[VALUE]
    if not isinstance(v, AbstractValue):
        raise MyiaTypeError('unsafe_static_cast expects a type constant')
    return v


@standard_prim(P.J)
async def _inf_J(engine, x):
    if isinstance(x, AbstractFunction):
        v = await x.get()
        return AbstractFunction(*[JTransformedFunction(poss)
                                  for poss in v])
    return AbstractJTagged(x)


@standard_prim(P.Jinv)
async def _inf_Jinv(engine, x):
    if isinstance(x, AbstractFunction):
        v = await x.get()
        results = []
        for f in v:
            if isinstance(f, JTransformedFunction):
                res = f.fn
            elif isinstance(f, GraphFunction):
                g = f.graph
                primal = g and g.transforms.get('primal', None)
                if primal:
                    primal = engine.pipeline.resources.convert(primal)
                    if isinstance(primal, Graph):
                        if primal.parent:
                            # The primal for a closure can't be used
                            # because it points to the original nodes
                            # of its parent, whereas we would like to
                            # point to the transformed nodes of the
                            # parent. This is fixable, and will need
                            # to be fixed to support a few edge cases.
                            res = DummyFunction()
                        else:
                            res = GraphFunction(primal, Context.empty())
                    else:
                        res = primal
                        if isinstance(res, Primitive):
                            tid = getattr(f, 'tracking_id', None)
                            res = PrimitiveFunction(res, tracking_id=tid)
                else:
                    raise MyiaTypeError(f'Bad input type for Jinv: {f}')
            else:
                raise MyiaTypeError(
                    f'Expected JTransformedFunction, not {f}'
                )
            results.append(res)
        return AbstractFunction(*results)
    if isinstance(x, AbstractJTagged):
        return x.element
    else:
        raise MyiaTypeError('Expected JTagged')
