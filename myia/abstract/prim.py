"""Inferrers for primitives."""

import operator
import inspect
from functools import reduce
from collections import defaultdict
from operator import getitem

from .. import dtype
from ..abstract import typecheck
from ..ir import Graph, MetaGraph, GraphCloner, CloneRemapper, new_graph
from ..dtype import Number, Bool, ExceptionType
from ..prim import ops as P, Primitive, py_implementations as py
from ..utils import Namespace, SymbolicKeyInstance, Cons, Empty, \
    MyiaTypeError, InferenceError, MyiaShapeError, check_nargs, \
    infer_trace, type_error_nargs

from .data import (
    ANYTHING,
    AbstractValue,
    AbstractScalar,
    AbstractType,
    AbstractFunction,
    AbstractTuple,
    AbstractArray,
    AbstractDict,
    AbstractClassBase,
    AbstractADT,
    AbstractJTagged,
    AbstractBottom,
    AbstractUnion,
    AbstractTaggedUnion,
    PartialApplication,
    JTransformedFunction,
    PrimitiveFunction,
    Possibilities,
    GraphFunction,
    MetaGraphFunction,
    DummyFunction,
    VALUE, TYPE, SHAPE,
    listof,
)
from .loop import Pending, find_coherent_result, force_pending
from .ref import Context
from .utils import sensitivity_transform, build_value, \
    type_token, broaden, type_to_abstract, hastype_helper
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
        self.nargs = None if data.varargs else len(data.args) - 2
        self.typemap = {}
        for name, ann in data.annotations.items():
            self.typemap[data.args.index(name) - 2] = ann

    async def infer(self, engine, *args):
        """Infer the abstract result given the abstract arguments."""
        check_nargs(self.prim, self.nargs, args)
        infer_trace.set({**infer_trace.get(), self.prim: (self.prim, args)})
        for i, arg in enumerate(args):
            typ = self.typemap.get(i)
            if typ is None:
                pass
            elif isinstance(typ, dtype.TypeMeta):
                await force_pending(engine.check(typ, type_token(arg)))
            elif isinstance(typ, type) and issubclass(typ, AbstractValue):
                if not isinstance(arg, typ):
                    raise MyiaTypeError(
                        f'Wrong type {arg} != {typ} for {self.prim}'
                    )
            elif callable(typ):
                await force_pending(engine.check(typ, arg))
        return await self._infer(self, engine, *args)

    def require_constant(self, a, *, argnum, range=None):
        """Returns the constant associated to abstract argument a.

        If a is not a constant, raises a MyiaTypeError.

        Arguments:
            argnum (int): Which argument we are checking.
            range (optional): A range or collection in which the argument
                must lie.
        """
        v = a.values[VALUE]
        if v is ANYTHING:
            raise MyiaTypeError(
                f'Argument {argnum} to {self.prim} must be constant.'
            )
        if range is not None and v not in range:
            raise MyiaTypeError(
                f'Argument {argnum} to {self.prim} is out of range.'
                f' It should lie in {range}'
            )
        return v


def standard_prim(prim):
    """Decorator to define and register a StandardInferrer."""
    def deco(fn):
        if isinstance(fn, type):
            abstract_inferrer_constructors[prim] = fn.partial()
        else:
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

    def normalize_args_sync(self, args):
        """If infer_value is False, return broadened arguments."""
        if not self.infer_value:
            args = tuple(broaden(a) for a in args)
        return args

    async def infer(self, engine, *args):
        """Infer the abstract result given the abstract arguments."""
        check_nargs(self.prim, self.nargs, args)
        infer_trace.set({**infer_trace.get(), self.prim: (self.prim, args)})
        outtype = self.outtype
        if any(not isinstance(arg, AbstractScalar) for arg in args):
            raise MyiaTypeError(f'Expected scalar as argument to {self.prim}')
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
    # Check that afn represents a single Primitive/Graph/MetaGraph
    # and return it.
    assert isinstance(afn, AbstractFunction)
    fn = afn.get_unique()
    if isinstance(fn, PrimitiveFunction):
        fn = fn.prim
    if isinstance(fn, GraphFunction):
        assert fn.context == Context.empty()
        fn = fn.graph
    if isinstance(fn, MetaGraphFunction):
        assert fn.context == Context.empty()
        fn = fn.metagraph
    assert isinstance(fn, (Primitive, Graph, MetaGraph))
    return fn


async def static_getter(engine, data, item, fetch, on_dcattr, chk=None,
                        dataref=None, outref=None):
    """Analyze a call to getattr or resolve.

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

    Returns:
        ('rval', value): An AbstractValue representing the result type
            of the call.
        ('reroute', ref): A new reference to reroute the call to.

    """
    resources = engine.pipeline.resources

    data_t = type_token(data)
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
        if item_v in data.attributes:
            return 'rval', await on_dcattr(data, item_v)
        elif item_v in data.methods:
            method = data.methods[item_v]
            isprop = isinstance(method, property)
            if isprop:
                method = resources.convert(method.fget)
            else:
                method = resources.convert(method)
            inferrer = to_abstract(method, Context.empty(), ref=outref)
            fn = _prim_or_graph(inferrer)
            g = outref.node.graph
            eng = outref.engine
            if isprop:
                ref = eng.ref(g.apply(fn, dataref.node),
                              outref.context)
            else:
                ref = eng.ref(g.apply(P.partial, fn, dataref.node),
                              outref.context)
            return 'reroute', ref
        else:
            raise InferenceError(f'Unknown field in {data}: {item_v}')

    elif case == 'method':
        method, = args
        method = resources.convert(method)
        inferrer = to_abstract(method, Context.empty(), ref=outref)
        fn = _prim_or_graph(inferrer)
        g = outref.node.graph
        eng = outref.engine
        ref = eng.ref(g.apply(P.partial, fn, dataref.node),
                      outref.context)
        return 'reroute', ref

    elif case == 'no_method':
        msg = f"object of type {data} has no attribute '{item_v}'"
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
        return 'rval', to_abstract(value, Context.empty(), ref=outref)


async def _resolve_case(resources, data_t, item_v, chk):
    mmap = resources.method_map

    if isinstance(data_t, type) and issubclass(data_t, AbstractClassBase):
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
uniform_prim(P.scalar_max)(py.scalar_max)
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


@standard_prim(P.typeof)
async def _inf_typeof(self, engine, value):
    return AbstractType(value)


@standard_prim(P.hastype)
async def _inf_hastype(self, engine, value, model: AbstractType):
    a = type_to_abstract(model.values[VALUE])
    return AbstractScalar({
        VALUE: hastype_helper(value, a),
        TYPE: dtype.Bool,
    })


###################
# Data structures #
###################


@standard_prim(P.make_tuple)
async def _inf_make_tuple(self, engine, *args):
    return AbstractTuple(args)


@standard_prim(P.make_list)
class _MakeListInferrer(Inferrer):
    async def reroute(self, engine, outref, argrefs):
        g = outref.node.graph
        lst = g.apply(Empty)
        argtypes = [await arg.get() for arg in argrefs]
        if argtypes == []:
            return engine.ref(lst, outref.context)
        restype = engine.abstract_merge(*argtypes)
        for arg in reversed(argrefs):
            lst = g.apply(Cons, arg.node, lst)
        rval = g.apply(P.unsafe_static_cast, lst, listof(restype))
        return engine.ref(rval, outref.context)


@standard_prim(P.make_dict)
async def _inf_make_dict(self, engine, _dct: AbstractType, *values):
    dct = _dct.values[VALUE]
    assert len(dct.entries) == len(values)
    for t, elem in zip(dct.entries.values(), values):
        assert typecheck(t, elem)

    return AbstractDict(dict((key, val) for key, val in
                             zip(dct.entries.keys(), values)))


@standard_prim(P.make_record)
async def infer_type_make_record(self, engine, _cls: AbstractType, *elems):
    """Infer the return type of make_record."""
    cls = _cls.values[VALUE]
    cls = type_to_abstract(cls)
    expected = list(cls.attributes.items())
    if len(expected) != len(elems):
        raise MyiaTypeError(
            f'{cls.tag.__qualname__} expects {len(expected)} fields '
            f'but got {len(elems)} instead.'
        )
    for (name, t), elem in zip(expected, elems):
        if not typecheck(t, elem):
            raise MyiaTypeError(
                f'{cls.tag.__qualname__} expects field `{name}` '
                f'to have type {elem} but got {t}'
            )

    wrap = broaden if type(cls) is AbstractADT else None

    return type(cls)(
        cls.tag,
        {
            name: wrap(elem) if wrap else elem
            for (name, _), elem in zip(expected, elems)
        },
        cls.methods,
        constructor=cls.constructor
    )


@standard_prim(P.tuple_getitem)
async def _inf_tuple_getitem(self, engine,
                             arg: AbstractTuple, idx: dtype.Int[64]):
    nelems = len(arg.elements)
    idx_v = self.require_constant(idx, argnum=2, range=range(-nelems, nelems))
    return arg.elements[idx_v]


@standard_prim(P.dict_getitem)
async def _inf_dict_getitem(self, engine, arg: AbstractDict, idx):
    idx_v = idx.values[VALUE]
    if idx_v is ANYTHING:
        raise MyiaTypeError(
            'Dictionaries must be indexed with a constant'
        )
    if not isinstance(idx_v, str):
        raise MyiaTypeError(
            f'Dictionary indexes must be strings, not {idx_v}.'
        )
    if idx_v not in arg.entries:
        raise MyiaTypeError(f'Invalid index for indexed dictionary')
    return arg.entries[idx_v]


@standard_prim(P.dict_values)
async def _inf_dict_values(self, engine, arg: AbstractDict):
    return AbstractTuple(list(arg.entries.values()))


@standard_prim(P.tuple_setitem)
async def _inf_tuple_setitem(self, engine,
                             arg: AbstractTuple,
                             idx: dtype.Int[64],
                             value: AbstractValue):
    nelems = len(arg.elements)
    idx_v = self.require_constant(idx, argnum=2, range=range(-nelems, nelems))
    elts = arg.elements
    new_elts = tuple([*elts[:idx_v], value, *elts[idx_v + 1:]])
    return AbstractTuple(new_elts)


def _getattr_chk(data_v, item_v):
    if not isinstance(item_v, str):  # pragma: no cover
        raise MyiaTypeError(
            f'Argument to getattr must be a string, not {item_v}.'
        )


async def _getattr_on_dcattr(data, item_v):
    return data.attributes[item_v]


@standard_prim(P.getattr)
class _GetAttrInferrer(Inferrer):
    async def reroute(self, engine, outref, argrefs):
        check_nargs(P.getattr, 2, argrefs)
        r_data, r_item = argrefs
        data = await r_data.get()

        if isinstance(data, AbstractUnion):
            g = outref.node.graph
            currg = g
            opts = await force_pending(data.options)
            for i, opt in enumerate(opts):
                last = (i == len(opts) - 1)
                if last:
                    falseg = None
                    cast = currg.apply(P.unsafe_static_cast, r_data.node, opt)
                    out = currg.apply(P.getattr, cast, r_item.node)
                else:
                    trueg = Graph()
                    falseg = Graph()
                    cond = currg.apply(P.hastype, r_data.node, opt)
                    cast = trueg.apply(P.unsafe_static_cast, r_data.node, opt)
                    trueg.output = trueg.apply(P.getattr, cast, r_item.node)
                    engine.mng.add_graph(trueg)
                    out = currg.apply(P.switch, cond, trueg, falseg)
                    out = currg.apply(out)
                if currg is g:
                    rval = out
                else:
                    currg.output = out
                    engine.mng.add_graph(currg)
                currg = falseg
            return engine.ref(rval, outref.context)

        else:
            item = await r_item.get()
            policy, newref = await static_getter(
                engine, data, item,
                fetch=getattr,
                on_dcattr=_getattr_on_dcattr,
                chk=_getattr_chk,
                outref=outref,
                dataref=r_data,
            )
            return newref if policy == 'reroute' else None

    async def run(self, engine, outref, argrefs):
        check_nargs(P.getattr, 2, argrefs)
        r_data, r_item = argrefs
        data = await r_data.get()
        item = await r_item.get()
        policy, rval = await static_getter(
            engine, data, item,
            fetch=getattr,
            on_dcattr=_getattr_on_dcattr,
            chk=_getattr_chk,
            outref=outref,
            dataref=r_data,
        )
        assert policy == 'rval'
        self.cache[(data, item)] = rval
        return rval


# TODO: setattr


@standard_prim(P.tuple_len)
async def _inf_tuple_len(self, engine, xs: AbstractTuple):
    return AbstractScalar({
        VALUE: len(xs.elements),
        TYPE: dtype.Int[64],
    })


@standard_prim(P.array_len)
async def _inf_array_len(self, engine, xs: AbstractArray):
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.Int[64],
    })


##########
# Arrays #
##########


@standard_prim(P.scalar_to_array)
async def _inf_scalar_to_array(self, engine, a: AbstractScalar, t):
    tp = t.values[VALUE]
    tp = type(tp)
    assert issubclass(tp, AbstractArray)
    return tp(a, {SHAPE: ()})


@standard_prim(P.array_to_scalar)
async def _inf_array_to_scalar(self, engine, a: AbstractArray):
    a_shp = a.values[SHAPE]
    if len(a_shp) != 0:
        raise MyiaShapeError("array_to_scalar requires shape ()")
    return a.element


@standard_prim(P.broadcast_shape)
async def _inf_broadcast_shape(self, engine, xs: _shape_type, ys: _shape_type):
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
async def _inf_invert_permutation(self, engine, perm: _shape_type):
    v = [x.values[VALUE] for x in perm.elements]
    return AbstractTuple(
        [perm.elements[i] if i in v else AbstractScalar({
            VALUE: ANYTHING,
            TYPE: dtype.UInt[64],
        }) for i in range(len(v))]
    )


@standard_prim(P.shape)
async def _inf_shape(self, engine, a: AbstractArray):
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
async def _inf_array_map(self, engine, fn: AbstractFunction, *arrays):
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

    for arr in arrays:
        if type(arrays[0]) != type(arr):
            raise MyiaTypeError(
                f'Expect array of type {type(arrays[0])} '
                f'to have same type as array of type {type(arr)}')

    return type(arrays[0])(result, {SHAPE: tuple(rshape)})


# TODO: array_scan


@standard_prim(P.array_reduce)
async def _inf_array_reduce(self, engine,
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
    return type(a)(res, {SHAPE: shp_v})


@standard_prim(P.distribute)
async def _inf_distribute(self, engine, a: AbstractArray, _shp: _shape_type):
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
    return type(a)(a.element, {SHAPE: shp})


@standard_prim(P.reshape)
async def _inf_reshape(self, engine, a: AbstractArray, _shp: _shape_type):
    shp = build_value(_shp, default=ANYTHING)
    if shp == ANYTHING:
        shp = (ANYTHING,) * len(_shp.elements)
    a_shp = await force_pending(a.values[SHAPE])
    if (all(s is not ANYTHING for s in shp) and
        all(s is not ANYTHING for s in a_shp) and
            prod(shp) != prod(a_shp)):
        raise MyiaShapeError("Cannot change the total number of elements "
                             "in reshape")
    return type(a)(a.element, {SHAPE: shp})


@standard_prim(P.transpose)
async def _inf_transpose(self, engine,
                         a: AbstractArray, permutation: _shape_type):
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
    return type(a)(a.element, {SHAPE: shp})


@standard_prim(P.dot)
async def _inf_dot(self, engine, a: AbstractArray, b: AbstractArray):
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

    if type(a) != type(b):
        raise MyiaTypeError(
            f'Expect array of type {type(a)} '
            f'to have same type as array of type {type(b)}')

    return type(a)(a.element, {SHAPE: c_shp})


##############
# Statements #
##############


class _CastRemapper(CloneRemapper):

    def __init__(self,
                 graphs,
                 inlines,
                 manager,
                 relation,
                 graph_relation,
                 clone_constants,
                 graph_repl,
                 fv_replacements):
        """Initialize the GraphCloner."""
        super().__init__(
            graphs=graphs,
            inlines=inlines,
            manager=manager,
            relation=relation,
            graph_repl=graph_repl,
            graph_relation=graph_relation,
            clone_constants=clone_constants,
        )
        self.fv_replacements = fv_replacements

    def gen_fv(self, g, ng, fv):
        """Remap the free variables we want to remap."""
        if fv in self.fv_replacements:
            new = self.fv_replacements[fv]
            self.remap_node((g, fv), g, fv, ng, new, link=False)


@standard_prim(P.user_switch)
class _UserSwitchInferrer(Inferrer):

    async def type_trials(self, engine, focus, outref,
                          opnode, argrefs,
                          condref, tbref, fbref):
        """Handle `user_switch(hastype(x, typ), tb, fb)`.

        We want to evaluate tb in a context where x has type typ and fb
        in a context where it doesn't.
        """
        def cond_trial(cg, opt):
            # For each possible type we make a "cond trial" which replaces the
            # focus input in the condition function by one that's cast to the
            # type. We can thus check if the value of the condition depends
            # directly on the type.
            return cg.apply(opnode,
                            *nodes[:focus],
                            cg.apply(P.unsafe_static_cast, nodes[focus], opt),
                            *nodes[focus + 1:])

        async def wrap(branch_ref, branch_type):
            # We transform branch_graph into a new graph which refers to a cast
            # version of x. We also transform all of the children of x's graph
            # so that closures called in the branch also refer to the cast
            # version of x.
            branch_graph = branch_ref.node.value
            if branch_graph not in xg.scope:
                return branch_graph
            rval = new_graph(branch_graph, relation='copy')
            cast = rval.apply(P.unsafe_static_cast, xref.node, branch_type)
            cl = GraphCloner(
                *xg.children,
                total=False,
                graph_repl={branch_graph: rval},
                remapper_class=_CastRemapper.partial(
                    fv_replacements={xref.node: cast}
                )
            )
            assert rval is cl[branch_graph]
            engine.mng.add_graph(rval)
            return rval

        nodes = [ref.node for ref in argrefs]
        xref = argrefs[focus]
        fulltype = await xref.get()
        assert isinstance(fulltype, AbstractUnion)

        xg = xref.node.graph
        cg = condref.node.graph
        cond_trials = [cond_trial(cg, t) for t in fulltype.options]
        results = [await engine.ref(node, condref.context).get()
                   for node in cond_trials]

        groups = {True: [], False: [], ANYTHING: []}

        for t, result in zip(fulltype.options, results):
            assert isinstance(result, AbstractScalar)
            assert result.values[TYPE] is dtype.Bool
            value = result.values[VALUE]
            groups[value].append(t)

        if groups[ANYTHING]:
            return await self.default(engine, outref, condref, tbref, fbref)

        from .utils import union_simplify
        from functools import reduce
        tbtyp = union_simplify(groups[True])
        fbtyp = union_simplify(groups[False])

        if tbtyp is None:
            return fbref
        elif fbtyp is None:
            return tbref
        else:
            g = outref.node.graph
            new_conds = [g.apply(P.hastype, xref.node, t)
                         for t in groups[True]]
            new_cond = reduce(lambda x, y: g.apply(P.bool_or, x, y),
                              new_conds)
            new_tb = await wrap(tbref, tbtyp)
            new_fb = await wrap(fbref, fbtyp)
            new_node = g.apply(P.switch, new_cond, new_tb, new_fb)
            return engine.ref(new_node, outref.context)

    async def _find_op(self, engine, condref):
        """Find a primitive operator to use for the condition."""
        ctx = condref.context
        if condref.node.is_apply():
            opnode, *args = condref.node.inputs
            opref = engine.ref(opnode, ctx)
            ops = (await opref.get()).get_sync()
            if len(ops) == 1:
                op, = ops
                argrefs = [engine.ref(a, ctx) for a in args]
                argtypes = [await arg.get() for arg in argrefs]
                for i, arg in enumerate(argtypes):
                    if isinstance(arg, AbstractUnion):
                        return i + 1, opnode, argrefs
        return None, None, None

    async def default(self, engine, outref, condref, tbref, fbref):
        g = outref.node.graph
        _, cond, tb, fb = outref.node.inputs
        condt = await condref.get()
        if not engine.check_predicate(Bool, type_token(condt)):
            to_bool = engine.pipeline.resources.convert(bool)
            cond = g.apply(to_bool, cond)
        newnode = g.apply(P.switch, cond, tb, fb)
        return engine.ref(newnode, outref.context)

    async def reroute(self, engine, outref, argrefs):
        check_nargs(P.switch, 3, argrefs)
        condref, tbref, fbref = argrefs

        for branch_ref in [tbref, fbref]:
            if not branch_ref.node.is_constant_graph():
                raise MyiaTypeError(
                    'Branches of switch must be functions when the condition'
                    ' is hastype on a Union.'
                )

        focus, opnode, args = await self._find_op(engine, condref)
        if focus is not None:
            return await self.type_trials(engine, focus - 1, outref,
                                          opnode, args, condref,
                                          tbref, fbref)

        return await self.default(engine, outref, condref, tbref, fbref)


@standard_prim(P.switch)
class _SwitchInferrer(Inferrer):

    async def run(self, engine, outref, argrefs):
        check_nargs(P.switch, 3, argrefs)
        condref, tbref, fbref = argrefs

        cond = await condref.get()
        await force_pending(engine.check(Bool, type_token(cond)))

        v = cond.values[VALUE]
        if v is True:
            return await tbref.get()
        elif v is False:
            return await fbref.get()
        elif v is ANYTHING:
            tb = await tbref.get()
            fb = await fbref.get()
            return engine.abstract_merge(tb, fb)
        else:
            raise AssertionError(f"Invalid condition value for switch: {v}")


#################
# Miscellaneous #
#################


@standard_prim(P.scalar_cast)
async def _inf_scalar_cast(self, engine,
                           scalar: Number,
                           typ: AbstractType):
    a = type_to_abstract(typ.values[VALUE])
    t = type_token(a)
    engine.check(Number, t)
    values = {**scalar.values, TYPE: t}
    return AbstractScalar(values)


@standard_prim(P.identity)
async def _inf_identity(self, engine, x):
    return x


@standard_prim(P.return_)
async def _inf_return(self, engine, x):
    return x


@standard_prim(P.raise_)
async def _inf_raise(self, engine, x: ExceptionType):
    return AbstractBottom()


@standard_prim(P.exception)
async def _inf_exception(self, engine, x):
    return AbstractScalar({VALUE: ANYTHING, TYPE: ExceptionType})


def _resolve_chk(data_v, item_v):
    if not isinstance(data_v, Namespace):  # pragma: no cover
        raise MyiaTypeError(
            f'data argument to resolve must be Namespace,'
            f' not {data_v}',
        )
    if not isinstance(item_v, str):  # pragma: no cover
        raise MyiaTypeError(
            f'item argument to resolve must be a string,'
            f' not {item_v}.',
        )


async def _resolve_on_dcattr(data, item_v):  # pragma: no cover
    raise MyiaTypeError('Cannot resolve on Class.')


@standard_prim(P.resolve)
class _ResolveInferrer(Inferrer):

    async def reroute(self, engine, outref, argrefs):
        check_nargs(P.resolve, 2, argrefs)
        r_data, r_item = argrefs
        data = await r_data.get()
        item = await r_item.get()
        policy, newref = await static_getter(
            engine, data, item,
            fetch=getitem,
            on_dcattr=_resolve_on_dcattr,
            chk=_resolve_chk,
            outref=outref,
            dataref=r_data,
        )
        return newref if policy == 'reroute' else None

    async def run(self, engine, outref, argrefs):
        check_nargs(P.resolve, 2, argrefs)
        r_data, r_item = argrefs
        data = await r_data.get()
        item = await r_item.get()
        policy, rval = await static_getter(
            engine, data, item,
            fetch=getitem,
            on_dcattr=_resolve_on_dcattr,
            chk=_resolve_chk,
            outref=outref,
            dataref=r_data,
        )
        assert policy == 'rval'
        self.cache[(data, item)] = rval
        return rval


@standard_prim(P.partial)
async def _inf_partial(self, engine, fn, *args):
    fns = await fn.get()
    assert isinstance(fns, Possibilities)
    return AbstractFunction(*[
        PartialApplication(fn, list(args)) for fn in fns
    ])


@standard_prim(P.apply)
class _ApplyInferrer(Inferrer):
    async def reroute(self, engine, outref, argrefs):
        assert len(argrefs) >= 1
        fnref, *grouprefs = argrefs
        fntyp = await fnref.get()
        expanded = []
        g = outref.node.graph
        kwinserts = []
        for gref in grouprefs:
            t = await gref.get()
            if isinstance(t, AbstractDict):
                fns = await fntyp.get()
                graphs = []
                for fn in fns:
                    if isinstance(fn, GraphFunction):
                        graphs.append(fn.graph)
                    else:
                        raise MyiaTypeError(
                            f'{fn} does not take keyword arguments'
                        )
                for k in t.entries:
                    try:
                        idxs = {graph.parameter_names.index(k)
                                for graph in graphs}
                    except ValueError:
                        raise MyiaTypeError(
                            'Invalid keyword argument'
                        )
                    if len(idxs) > 1:
                        raise MyiaTypeError(
                            'Inconsistent keyword argument positions'
                        )
                    idx, = idxs
                    extract = g.apply(P.dict_getitem, gref.node, k)
                    kwinserts.append((idx, extract))
            elif isinstance(t, AbstractTuple):
                for i, _ in enumerate(t.elements):
                    expanded.append(g.apply(P.tuple_getitem, gref.node, i))
            else:
                raise MyiaTypeError(
                    'Can only expand tuple or dict in function application'
                )
        for idx, node in kwinserts:
            expanded.insert(idx, node)
        return engine.ref(g.apply(fnref.node, *expanded), outref.context)


@standard_prim(P.embed)
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


@standard_prim(P.env_getitem)
async def _inf_env_getitem(self, engine,
                           env: dtype.EnvType,
                           key: dtype.SymbolicKeyType,
                           dflt):
    expected = key.values[VALUE].abstract
    engine.abstract_merge(expected, dflt)
    return expected


@standard_prim(P.env_setitem)
async def _inf_env_setitem(self, engine,
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
async def _inf_env_add(self, engine, env1, env2):
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.EnvType,
    })


@standard_prim(P.unsafe_static_cast)
async def _inf_unsafe_static_cast(self, engine, x, typ: AbstractType):
    return typ.values[VALUE]


@standard_prim(P.tagged)
async def _inf_tagged(self, engine, x, *rest):
    if len(rest) == 0:
        return AbstractUnion([broaden(x, engine.loop)])
    elif len(rest) == 1:
        tag, = rest
        tag_v = self.require_constant(tag, argnum=2)
        return AbstractTaggedUnion([[tag_v, broaden(x, engine.loop)]])
    else:
        raise type_error_nargs(P.tagged, "1 or 2", len(rest) + 1)


@standard_prim(P.hastag)
async def _inf_hastag(self, engine,
                      x: AbstractTaggedUnion, tag: dtype.Int[64]):
    opts = await force_pending(x.options)
    self.require_constant(
        tag, argnum=2,
        range={i for i, _ in opts}
    )
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: dtype.Bool,
    })


@standard_prim(P.casttag)
async def _inf_casttag(self, engine,
                       x: AbstractTaggedUnion, tag: dtype.Int[64]):
    opts = await force_pending(x.options)
    tag_v = self.require_constant(
        tag, argnum=2,
        range={i for i, _ in opts}
    )
    for i, typ in opts:
        if i == tag_v:
            return typ
    raise AssertionError('Unreachable')


@standard_prim(P.J)
async def _inf_J(self, engine, x):
    if isinstance(x, AbstractFunction):
        v = await x.get()
        return AbstractFunction(*[JTransformedFunction(poss)
                                  for poss in v])
    return AbstractJTagged(x)


@standard_prim(P.Jinv)
async def _inf_Jinv(self, engine, x):
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
                    raise MyiaTypeError(f'Bad input type for {self.prim}: {f}')
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
