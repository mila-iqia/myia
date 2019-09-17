"""Inferrers for primitives."""

import inspect
import operator
from collections import defaultdict
from functools import reduce

from .. import xtype
from ..ir import Graph
from ..prim import Primitive, ops as P, py_implementations as py
from ..utils import (
    MyiaAttributeError,
    MyiaShapeError,
    MyiaTypeError,
    check_nargs,
    infer_trace,
    type_error_nargs,
)
from ..xtype import Bool, ExceptionType, Number, String
from .data import (
    ANYTHING,
    SHAPE,
    TYPE,
    VALUE,
    AbstractADT,
    AbstractArray,
    AbstractBottom,
    AbstractClassBase,
    AbstractDict,
    AbstractFunction,
    AbstractJTagged,
    AbstractKeywordArgument,
    AbstractScalar,
    AbstractTaggedUnion,
    AbstractTuple,
    AbstractType,
    AbstractUnion,
    AbstractValue,
    DummyFunction,
    GraphFunction,
    JTransformedFunction,
    PartialApplication,
    Possibilities,
    PrimitiveFunction,
)
from .infer import Inferrer
from .loop import force_pending
from .ref import Context
from .utils import (
    broaden,
    build_value,
    hastype_helper,
    type_to_abstract,
    typecheck,
)

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
            elif isinstance(typ, xtype.TypeMeta):
                await force_pending(engine.check(typ, arg.xtype(), typ))
            elif isinstance(typ, type) and issubclass(typ, AbstractValue):
                if not isinstance(arg, typ):
                    raise MyiaTypeError(
                        f'Wrong type {arg} != {typ} for {self.prim}'
                    )
            elif callable(typ):
                await force_pending(engine.check(typ, arg))
            else:
                raise AssertionError(f'Invalid annotation: {typ}')
        return await self._infer(self, engine, *args)

    def require_constant(self, a, *, argnum, range=None):
        """Returns the constant associated to abstract argument a.

        If a is not a constant, raises a MyiaTypeError.

        Arguments:
            argnum (int): Which argument we are checking.
            range (optional): A range or collection in which the argument
                must lie.
        """
        v = a.xvalue()
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
            values = [arg.xvalue() for arg in args]
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
        ts = [arg.xtype() for arg in args]
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


def _shape_type(engine, shp):
    shp_t = engine.check(AbstractTuple, shp)
    for elem_t in shp_t.elements:
        engine.abstract_merge(xtype.UInt[64], elem_t.xtype())
    return shp_t


def _shape_type_pair(engine, shp):
    shp_t = _shape_type(engine, shp)
    if len(shp_t.elements) != 2:
        raise MyiaTypeError(f'Expected Tuple Length 2, not Tuple Length'
                            f'{len(shp_t.elements)}')
    return shp_t


def _stride_type(engine, shp):
    shp_t = engine.check(AbstractTuple, shp)
    for elem_t in shp_t.elements:
        engine.abstract_merge(xtype.Int[64], elem_t.values[TYPE])
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
uniform_prim(P.string_eq, infer_value=True)(py.string_eq)


######################
# Type introspection #
######################


@standard_prim(P.hastype)
async def _inf_hastype(self, engine, value, model: AbstractType):
    a = type_to_abstract(model.xvalue())
    return AbstractScalar({
        VALUE: hastype_helper(value, a),
        TYPE: xtype.Bool,
    })


###################
# Data structures #
###################


@standard_prim(P.make_tuple)
async def _inf_make_tuple(self, engine, *args):
    return AbstractTuple(args)


@standard_prim(P.make_dict)
async def _inf_make_dict(self, engine, _dct: AbstractType, *values):
    dct = _dct.xvalue()
    assert len(dct.entries) == len(values)
    for t, elem in zip(dct.entries.values(), values):
        assert typecheck(t, elem)

    return AbstractDict(dict((key, val) for key, val in
                             zip(dct.entries.keys(), values)))


@standard_prim(P.make_record)
async def _inf_make_record(self, engine, _cls: AbstractType, *elems):
    """Infer the return type of make_record."""
    cls = _cls.xvalue()
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
        constructor=cls.constructor
    )


@standard_prim(P.tuple_getitem)
async def _inf_tuple_getitem(self, engine,
                             arg: AbstractTuple, idx: xtype.Int[64]):
    nelems = len(arg.elements)
    idx_v = self.require_constant(idx, argnum=2, range=range(-nelems, nelems))
    return arg.elements[idx_v]


@standard_prim(P.tuple_setitem)
async def _inf_tuple_setitem(self, engine,
                             arg: AbstractTuple,
                             idx: xtype.Int[64],
                             value: AbstractValue):
    nelems = len(arg.elements)
    idx_v = self.require_constant(idx, argnum=2, range=range(-nelems, nelems))
    elts = arg.elements
    new_elts = tuple([*elts[:idx_v], value, *elts[idx_v + 1:]])
    return AbstractTuple(new_elts)


@standard_prim(P.dict_getitem)
async def _inf_dict_getitem(self, engine,
                            arg: AbstractDict,
                            idx: String):
    idx_v = self.require_constant(idx, argnum=2, range=set(arg.entries.keys()))
    return arg.entries[idx_v]


@standard_prim(P.dict_setitem)
async def _inf_dict_setitem(self, engine,
                            arg: AbstractDict,
                            idx: String,
                            value):
    idx_v = self.require_constant(idx, argnum=2, range=set(arg.entries.keys()))
    return type(arg)({**arg.entries, idx_v: value})


@standard_prim(P.record_getitem)
async def _inf_record_getitem(self, engine,
                              data: AbstractClassBase,
                              attr: String):
    attr_v = self.require_constant(attr, argnum=2)
    return data.attributes[attr_v]


@standard_prim(P.record_setitem)
async def _inf_record_setitem(self, engine,
                              data: AbstractClassBase,
                              attr: String,
                              value):
    attr_v = self.require_constant(attr, argnum=2)
    if attr_v not in data.attributes:
        raise MyiaAttributeError(f'Unknown field in {data}: {attr_v}')
    model = data.user_defined_version()
    expected = model.attributes[attr_v]
    if not typecheck(expected, value):
        raise MyiaTypeError(f'Expected field {attr_v} to have type {expected}')
    return type(data)(
        data.tag,
        {**data.attributes, attr_v: value},
        constructor=data.constructor
    )


##########
# Arrays #
##########


def _ceildiv(x, y):
    return -(-x // y)


@standard_prim(P.array_getitem)
async def _inf_array_getitem(self, engine, a: AbstractArray,
                             begin: _shape_type, end: _shape_type,
                             strides: _stride_type):

    begin = tuple(self.require_constant(e, argnum=f'"1:begin[{edx}]"')
                  for edx, e in enumerate(begin.elements))
    end = tuple(self.require_constant(e, argnum=f'"2:end[{edx}]"')
                for edx, e in enumerate(end.elements))
    strides = tuple(self.require_constant(e, argnum=f'"3:strides[{edx}]"')
                    for edx, e in enumerate(strides.elements))

    shp_before_stride = map(operator.sub, end, begin)
    shp = tuple(map(_ceildiv, shp_before_stride, map(abs, strides)))

    return type(a)(a.element, {SHAPE: shp, TYPE: a.xtype()})


@standard_prim(P.array_setitem)
async def _inf_array_setitem(self, engine, a: AbstractArray,
                             begin: _shape_type, end: _shape_type,
                             strides: _stride_type, value: AbstractArray):
    return a


@standard_prim(P.scalar_to_array)
async def _inf_scalar_to_array(self, engine, a: AbstractScalar, t):
    tp = t.xvalue()
    assert isinstance(tp, AbstractArray)
    return AbstractArray(a, {SHAPE: (), TYPE: tp.xtype()})


@standard_prim(P.array_to_scalar)
async def _inf_array_to_scalar(self, engine, a: AbstractArray):
    a_shp = a.xshape()
    if len(a_shp) != 0:
        raise MyiaShapeError("array_to_scalar requires shape ()")
    return a.element


@standard_prim(P.broadcast_shape)
async def _inf_broadcast_shape(self, engine, xs: _shape_type, ys: _shape_type):
    shp_x = tuple(x.xvalue() for x in xs.elements)
    shp_y = tuple(y.xvalue() for y in ys.elements)
    elems = []
    try:
        res = py.broadcast_shape(shp_x, shp_y)
    except ValueError as e:
        raise MyiaShapeError(e.args[0])
    for n in res:
        elems.append(AbstractScalar({
            VALUE: n,
            TYPE: xtype.UInt[64],
        }))
    return AbstractTuple(elems)


@standard_prim(P.invert_permutation)
async def _inf_invert_permutation(self, engine, perm: _shape_type):
    v = [x.xvalue() for x in perm.elements]
    return AbstractTuple(
        [perm.elements[i] if i in v else AbstractScalar({
            VALUE: ANYTHING,
            TYPE: xtype.UInt[64],
        }) for i in range(len(v))]
    )


@standard_prim(P.shape)
async def _inf_shape(self, engine, a: AbstractArray):
    shp = await force_pending(a.xshape())
    values = [
        AbstractScalar({
            VALUE: entry,
            TYPE: xtype.UInt[64],
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

    shapes = [a.xshape() for a in arrays]
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
        if arrays[0].xtype() != arr.xtype():
            raise MyiaTypeError(
                f'Expect array of type {arrays[0].xtype()} '
                f'to have same type as array of type {arr.xtype()}')

    return type(arrays[0])(
        result, {
            SHAPE: tuple(rshape),
            TYPE: arrays[0].xtype(),
        }
    )


# TODO: array_scan


@standard_prim(P.array_reduce)
async def _inf_array_reduce(self, engine,
                            fn: AbstractFunction,
                            a: AbstractArray,
                            shp: _shape_type):

    shp_i = await force_pending(a.xshape())
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
    return type(a)(res, {SHAPE: shp_v, TYPE: a.xtype()})


@standard_prim(P.distribute)
async def _inf_distribute(self, engine, a: AbstractArray, _shp: _shape_type):
    shp = tuple(x.xvalue() for x in _shp.elements)
    a_shp = await force_pending(a.xshape())
    delta = len(shp) - len(a_shp)
    if delta < 0:
        raise MyiaShapeError("Cannot distribute to smaller shape")
    elif delta > 0:
        a_shp = (1,) * delta + a_shp
    for vs, s in zip(a_shp, shp):
        if vs != s and vs not in (1, ANYTHING) and s not in (1, ANYTHING):
            raise MyiaShapeError("Cannot change shape when distributing")
    return type(a)(a.element, {SHAPE: shp, TYPE: a.xtype()})


@standard_prim(P.reshape)
async def _inf_reshape(self, engine, a: AbstractArray, _shp: _shape_type):
    shp = build_value(_shp, default=ANYTHING)
    if shp == ANYTHING:
        shp = (ANYTHING,) * len(_shp.elements)
    a_shp = await force_pending(a.xshape())
    if (all(s is not ANYTHING for s in shp) and
        all(s is not ANYTHING for s in a_shp) and
            prod(shp) != prod(a_shp)):
        raise MyiaShapeError("Cannot change the total number of elements "
                             "in reshape")
    return type(a)(a.element, {SHAPE: shp, TYPE: a.xtype()})


@standard_prim(P.transpose)
async def _inf_transpose(self, engine,
                         a: AbstractArray, permutation: _shape_type):
    perm = build_value(permutation, default=ANYTHING)
    if perm == ANYTHING:
        shp = (ANYTHING,) * len(permutation.elements)
    else:
        a_shp = await force_pending(a.xshape())
        if list(sorted(perm)) != list(range(len(a_shp))):
            raise MyiaShapeError(
                'The second argument of transpose must be a permutation of'
                ' all of the array\'s axes.',
            )

        shp = tuple(a_shp[i] for i in perm)
    return type(a)(a.element, {SHAPE: shp, TYPE: a.xtype()})


@standard_prim(P.gather)
async def _inf_gather(self, engine, input, dim, index):
    return type(input)(
        input.element,
        {SHAPE: index.xshape(), TYPE: input.xtype()}
    )


@standard_prim(P.scatter)
async def _inf_scatter(self, engine, input, dim, index, src):
    return input


@standard_prim(P.scatter_add)
async def _inf_scatter_add(self, engine, input, dim, index, src):
    return input


@standard_prim(P.argmax)
async def _inf_argmax(self, engine, input, dim):
    shp = ()
    if dim.xvalue() is None:
        pass
    else:
        shp_inp = input.xshape()
        for sdx, s in enumerate(shp_inp):
            if sdx == dim.xvalue():
                shp = shp + (1,)
            else:
                shp = shp + (s,)
    return type(input)(
        AbstractScalar({VALUE: ANYTHING, TYPE: xtype.Int[64]}),
        {SHAPE: shp, TYPE: input.xtype()}
    )


@standard_prim(P.array_max)
async def _inf_array_max(self, engine, input, dim):
    shp = ()
    shp_inp = input.xshape()
    for sdx, s in enumerate(shp_inp):
        if sdx == dim.xvalue():
            shp = shp + (1,)
        else:
            shp = shp + (s,)
    return type(input)(input.element, {SHAPE: shp, TYPE: input.xtype()})


@standard_prim(P.dot)
async def _inf_dot(self, engine, a: AbstractArray, b: AbstractArray):
    a_shp = a.xshape()
    b_shp = b.xshape()
    if len(a_shp) != 2 or len(b_shp) != 2:
        raise MyiaShapeError("dot needs matrix inputs")
    if (a_shp[1] != b_shp[0] and
            a_shp[1] is not ANYTHING and b_shp[0] is not ANYTHING):
        raise MyiaShapeError(
            f"Incompatible shapes in dot: {a_shp} and {b_shp}"
        )
    engine.abstract_merge(a.element, b.element)
    c_shp = (a_shp[0], b_shp[1])

    if a.xtype() != b.xtype():
        raise MyiaTypeError(
            f'Expect array of type {a.xtype()} '
            f'to have same type as array of type {b.xtype()}')

    return type(a)(a.element, {SHAPE: c_shp, TYPE: a.xtype()})


@standard_prim(P.conv2d)
async def _inf_conv2d(self, engine, input: AbstractArray,
                      weight: AbstractArray, stride: _shape_type_pair,
                      padding: _shape_type_pair, dilation: _shape_type_pair,
                      groups: xtype.UInt[64]):

    # TODO: _shape_type should not allow float to be converted to uint
    # TODO: "groups: UInt[64]" should not allow float to be converted to uint

    h_in, w_in = input.xshape()[2:]
    kernel_size = weight.xshape()[2:]

    stride = tuple(self.require_constant(e, argnum=f'"2:stride[{edx}]"')
                   for edx, e in enumerate(stride.elements))
    padding = tuple(self.require_constant(e, argnum=f'"3:padding[{edx}]"')
                    for edx, e in enumerate(padding.elements))
    dilation = tuple(self.require_constant(e, argnum=f'"4:dilation[{edx}]"')
                     for edx, e in enumerate(dilation.elements))

    N = input.xshape()[0]
    C_out = weight.xshape()[0]

    # Based on formulae in shape section of:
    # https://pytorch.org/docs/stable/nn.html#conv2d
    H_out = ((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
             // stride[0]) + 1
    W_out = ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
             // stride[1]) + 1

    out_shape = (N, C_out, int(H_out), int(W_out))

    # Checks all elements of input have same xtype as all elements of weight
    engine.check(AbstractScalar, input.element, weight.element)
    # ^ TODO: PyTorch also enforces, but might want to change for mixed precis

    return type(weight)(weight.element, {SHAPE: out_shape,
                                         TYPE: weight.xtype()})


@standard_prim(P.conv2d_input_grad)
async def _inf_conv2d_input_grad(self, engine, input_size: _shape_type,
                                 weight: AbstractArray,
                                 grad_output: AbstractArray,
                                 stride: _shape_type_pair,
                                 padding: _shape_type_pair,
                                 dilation: _shape_type_pair,
                                 groups: xtype.UInt[64]):
    input_size_tuple = tuple(
        self.require_constant(i_s, argnum=0) for i_s in input_size.elements)
    return type(weight)(weight.element, {SHAPE: input_size_tuple,
                                         TYPE: weight.xtype()})


@standard_prim(P.conv2d_weight_grad)
async def _inf_conv2d_weight_grad(self, engine, input: AbstractArray,
                                  weight_size: _shape_type,
                                  grad_output: AbstractArray,
                                  stride: _shape_type_pair,
                                  padding: _shape_type_pair,
                                  dilation: _shape_type_pair,
                                  groups: xtype.UInt[64]):
    weight_size_tuple = tuple(
        self.require_constant(w_s, argnum=0) for w_s in weight_size.elements)
    return type(input)(input.element, {SHAPE: weight_size_tuple,
                                       TYPE: input.xtype()})


@standard_prim(P.max_pool2d)
async def _inf_max_pool2d(self, engine, input, kernel_size, stride, padding,
                          dilation, ceil_mode):

    # TODO: _shape_type should not allow float to be converted to uint

    h_in, w_in = input.xshape()[2:]

    kernel_size = tuple(self.require_constant(
                        e, argnum=f'"1:kernel_size[{edx}]"')
                        for edx, e in enumerate(kernel_size.elements))
    stride = tuple(self.require_constant(e, argnum=f'"2:stride[{edx}]"')
                   for edx, e in enumerate(stride.elements))
    padding = tuple(self.require_constant(e, argnum=f'"3:padding[{edx}]"')
                    for edx, e in enumerate(padding.elements))
    dilation = tuple(self.require_constant(e, argnum=f'"4:dilation[{edx}]"')
                     for edx, e in enumerate(dilation.elements))

    N = input.xshape()[0]
    C_out = input.xshape()[1]

    # Based on formulae in shape section of:
    # https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d
    H_out = ((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
             // stride[0]) + 1
    W_out = ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
             // stride[1]) + 1

    out_shape = (N, C_out, int(H_out), int(W_out))

    return AbstractTuple([
        type(input)(input.element, {SHAPE: out_shape, TYPE: input.xtype()}),
        type(input)(AbstractScalar({VALUE: ANYTHING, TYPE: xtype.Int[64]}),
                    {SHAPE: out_shape, TYPE: input.xtype()}
                    )
    ])


@standard_prim(P.max_pool2d_grad)
async def _inf_max_pool2d_grad(self, engine, input, kernel_size, stride,
                               padding, dilation, ceil_mode, dout):
    return input


##############
# Statements #
##############


@standard_prim(P.switch)
class _SwitchInferrer(Inferrer):

    async def run(self, engine, outref, argrefs):
        condref, tbref, fbref = check_nargs(P.switch, 3, argrefs)

        cond = await condref.get()
        await force_pending(engine.check(Bool, cond.xtype()))

        v = cond.xvalue()
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
    a = type_to_abstract(typ.xvalue())
    t = a.xtype()
    engine.check(Number, t)
    values = {**scalar.values, TYPE: t}
    return AbstractScalar(values)


@standard_prim(P.array_cast)
async def _inf_array_cast(self, engine,
                          a: AbstractArray,
                          typ: AbstractType):
    t = (type_to_abstract(typ.xvalue())).xtype()
    engine.check(Number, t)
    e_values = {**a.element.values, TYPE: t}
    return AbstractArray(AbstractScalar(e_values), a.values)


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


@standard_prim(P.partial)
async def _inf_partial(self, engine, fn, *args):
    fns = await fn.get()
    assert isinstance(fns, Possibilities)
    return AbstractFunction(*[
        PartialApplication(fn, list(args)) for fn in fns
    ])


@standard_prim(P.make_kwarg)
async def _inf_make_kwarg(self, engine, key, value):
    k = key.xvalue()
    assert isinstance(k, str)
    return AbstractKeywordArgument(k, value)


@standard_prim(P.extract_kwarg)
class _ExtractKwArgInferrer(Inferrer):
    async def normalize_args(self, args):
        return args

    async def infer(self, engine, key, kwarg):
        assert key.xvalue() is kwarg.key
        return kwarg.argument


@standard_prim(P.env_getitem)
async def _inf_env_getitem(self, engine,
                           env: xtype.EnvType,
                           key: xtype.SymbolicKeyType,
                           dflt):
    expected = key.xvalue().abstract
    engine.abstract_merge(expected, dflt)
    return expected


@standard_prim(P.env_setitem)
async def _inf_env_setitem(self, engine,
                           env: xtype.EnvType,
                           key: xtype.SymbolicKeyType,
                           value):
    expected = key.xvalue().abstract
    engine.abstract_merge(expected, value)
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: xtype.EnvType,
    })


@standard_prim(P.env_add)
async def _inf_env_add(self, engine, env1, env2):
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: xtype.EnvType,
    })


@standard_prim(P.unsafe_static_cast)
async def _inf_unsafe_static_cast(self, engine, x, typ: AbstractType):
    return typ.xvalue()


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
                      x: AbstractTaggedUnion, tag: xtype.Int[64]):
    opts = await force_pending(x.options)
    self.require_constant(
        tag, argnum=2,
        range={i for i, _ in opts}
    )
    return AbstractScalar({
        VALUE: ANYTHING,
        TYPE: xtype.Bool,
    })


@standard_prim(P.casttag)
async def _inf_casttag(self, engine,
                       x: AbstractTaggedUnion, tag: xtype.Int[64]):
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
                    primal = engine.resources.convert(primal)
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
