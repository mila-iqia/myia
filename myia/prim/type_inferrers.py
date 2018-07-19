"""Definitions of type inference for primitives."""


from functools import partial

from ..dtype import Int, Bool, Float, Tuple, List, Type, Array, UInt, Number
from ..infer import ANYTHING, Inferrer, GraphInferrer, PartialInferrer, \
    MyiaTypeError, register_inferrer, Track
from ..ir import Graph
from ..utils import Namespace

from . import ops as P
from .inferrer_utils import static_getter
from .ops import Primitive
from .py_implementations import typeof


type_inferrer_constructors = {}


class TypeTrack(Track):
    """Infer the type of a constant.

    Note: the type of a Primitive or of a Graph is an Inferrer.

    Attributes:
        constructors: A map of Inferrer constructors. Each constructor
            takes an engine as argument and returns an Inferrer. These
            will be used to infer types for primitives.

    """

    def __init__(self,
                 engine,
                 name,
                 *,
                 constructors=type_inferrer_constructors):
        """Initialize a TypeTrack.

        If a TypeTrack is present, it is always required.
        """
        super().__init__(engine, name)
        self.constructors = constructors

    def from_value(self, v, context):
        """Infer the type of a constant."""
        if isinstance(v, Primitive):
            return self.constructors[v](self)
        elif isinstance(v, Graph):
            return GraphInferrer(self, v, context)
        else:
            return typeof(v)

    def default(self):
        """Return a default type; this method raises an exception."""
        raise Exception('There is no default value for the type track.') \
            # pragma: no cover


########################
# Default constructors #
########################


type_inferrer = partial(register_inferrer,
                        constructors=type_inferrer_constructors)


@type_inferrer(P.if_, nargs=3)
async def infer_type_if(track, cond, tb, fb):
    """Infer the return type of if."""
    cond_t = await cond['type']
    if cond_t != Bool():
        raise MyiaTypeError('Condition for if must be a boolean')
    tb_inf = await tb['type']
    fb_inf = await fb['type']
    if not isinstance(tb_inf, Inferrer) or not isinstance(fb_inf, Inferrer):
        raise MyiaTypeError('Both branches of if primitive must be thunks') \
            # pragma: no cover
    v = await cond['value']
    if v is True:
        # We only visit the first branch if the condition is provably true
        return await tb_inf()
    elif v is False:
        # We only visit the second branch if the condition is provably false
        return await fb_inf()
    elif v is ANYTHING:
        # The first branch to finish will return immediately. When the other
        # branch finishes, its result will be checked against the other.
        return await track.assert_same(tb_inf(), fb_inf())
    else:
        raise AssertionError("Invalid condition value for if")


@type_inferrer(P.partial, nargs=None)
async def infer_type_partial(track, fn, *args):
    """Infer the return type of partial."""
    fn_t = await fn['type']
    return PartialInferrer(track, fn_t, args)


@type_inferrer(P.cons_tuple, nargs=2)
async def infer_type_cons_tuple(track, x, y):
    """Infer the return type of cons_tuple."""
    x_t = await x['type']
    y_t = await y['type']
    if not isinstance(y_t, Tuple):
        raise MyiaTypeError('cons_tuple on non-tuple')  # pragma: no cover
    return Tuple([x_t, *y_t.elements])


@type_inferrer(P.head, nargs=1)
async def infer_type_head(track, tup):
    """Infer the return type of head."""
    tup_t = await tup['type']
    if not isinstance(tup_t, Tuple):
        raise MyiaTypeError('head of non-tuple')
    if not len(tup_t.elements) >= 1:
        raise MyiaTypeError('head on empty tuple')
    return tup_t.elements[0]


@type_inferrer(P.tail, nargs=1)
async def infer_type_tail(track, tup):
    """Infer the return type of tail."""
    tup_t = await tup['type']
    if not isinstance(tup_t, Tuple):
        raise MyiaTypeError('tail of non-tuple')
    if not len(tup_t.elements) >= 1:
        raise MyiaTypeError('tail on empty tuple')
    return Tuple(tup_t.elements[1:])


@type_inferrer(P.getitem, nargs=2)
async def infer_type_getitem(track, seq, idx):
    """Infer the return type of getitem."""
    seq_t = await seq['type']
    idx_t = await idx['type']
    if not isinstance(idx_t, Int):
        raise MyiaTypeError('Expected Int for index')

    if isinstance(seq_t, Tuple):
        idx_v = await idx['value']
        if idx_v is ANYTHING:
            raise MyiaTypeError('Tuples must be indexed with a constant')
        return seq_t.elements[idx_v]
    elif isinstance(seq_t, List):
        return seq_t.element_type
    else:
        raise MyiaTypeError('Wrong seq type for getitem')


@type_inferrer(P.typeof, nargs=1)
async def infer_type_typeof(track, _):
    """Infer the return type of typeof."""
    return Type


@type_inferrer(P.hastype, nargs=2)
async def infer_type_hastype(track, x, t):
    """Infer the return type of hastype."""
    t_t = await t['type']
    if t_t is not Type:
        raise MyiaTypeError(
            f'Second argument to hastype must be a Type, got {t_t}'
        )
    return Bool()


@type_inferrer(P.bool_not, nargs=1)
async def infer_type_bool_not(track, x):
    """Infer the return type of not."""
    x_t = await x['type']
    if x_t != Bool():
        raise MyiaTypeError('Expected Bool for not.')
    return Bool()


@type_inferrer(P.bool_and, nargs=2)
async def infer_type_bool_and(track, x, y):
    """Infer the return type of bool_and."""
    x_t = await x['type']
    y_t = await y['type']
    if x_t != Bool() or y_t != Bool():
        raise MyiaTypeError('Expected Bool for bool_and.')
    return Bool()


@type_inferrer(P.bool_or, nargs=2)
async def infer_type_bool_or(track, x, y):
    """Infer the return type of bool_or."""
    x_t = await x['type']
    y_t = await y['type']
    if x_t != Bool() or y_t != Bool():
        raise MyiaTypeError('Expected Bool for bool_or.')
    return Bool()


@type_inferrer(P.scalar_eq, P.scalar_ne, nargs=2)
async def infer_type_generic_compare(track, x, y):
    """Infer the return type of a generic comparison operator."""
    await track.assert_same(x, y)
    return Bool()


@type_inferrer(P.scalar_lt, P.scalar_gt, P.scalar_le, P.scalar_ge, nargs=2)
async def infer_type_arith_compare(track, x, y):
    """Infer the return type of an arithmetic comparison operator."""
    t = await track.assert_same(x, y)
    if not isinstance(t, (Int, Float)):
        raise MyiaTypeError('Expected number')
    return Bool()


@type_inferrer(P.scalar_uadd, P.scalar_usub, nargs=1)
async def infer_type_arith_unary(track, x):
    """Infer the return type of a unary arithmetic operator."""
    t = await x['type']
    if not isinstance(t, (Int, Float)):
        raise MyiaTypeError('Expected number')
    return t


@type_inferrer(P.scalar_add, P.scalar_sub, P.scalar_mul, P.scalar_div,
               P.scalar_mod, P.scalar_pow, nargs=2)
async def infer_type_arith_bin(track, x, y):
    """Infer the return type of a binary arithmetic operator."""
    t = await track.assert_same(x, y)
    if not isinstance(t, (Int, Float)):
        raise MyiaTypeError('Expected number')
    return t


@type_inferrer(P.shape, nargs=1)
async def infer_type_shape(track, ary):
    """Infer the return type of shape."""
    shp = await ary['shape']
    return Tuple([UInt(64)]*len(shp))


@type_inferrer(P.array_map, nargs=2)
async def infer_type_array_map(track, fn, ary):
    """Infer the return type of array_map."""
    fn_t = await fn['type']
    ary_t = await ary['type']
    if not isinstance(ary_t, Array):
        raise MyiaTypeError('Expected array')
    xref = track.engine.vref({'type': ary_t.elements})
    return Array(await fn_t(xref))


@type_inferrer(P.array_scan, P.array_reduce, nargs=4)
async def infer_type_across_array(track, fn, init, ary, ax):
    """Infer the return type of scan/array_reduce."""
    fn_t = await fn['type']
    ary_t = await ary['type']
    init_t = await init['type']
    ax_t = await ax['type']
    if not isinstance(ary_t, Array):
        raise MyiaTypeError('Expected array')
    if not ary_t.elements == init_t:
        raise MyiaTypeError("Initial value must have the same type "
                            "as array elements")
    if not ax_t == UInt(64):
        raise MyiaTypeError("Axis must be u64")
    xref = track.engine.vref({'type': ary_t.elements})
    xref2 = track.engine.vref({'type': ary_t.elements})
    return Array(await fn_t(xref, xref2))


@type_inferrer(P.distribute, nargs=2)
async def infer_type_distribute(track, v, shp):
    """Infer the return type of distribute."""
    v_t = await v['type']
    if isinstance(v_t, Array):
        v_t = v_t.elements
    elif not isinstance(v_t, (Number, Bool)):
        raise MyiaTypeError("Array elements must be numbers or bool")
    shp_t = await shp['type']
    if (not isinstance(shp_t, Tuple) or
            not all(e == UInt(64) for e in shp_t.elements)):
        raise MyiaTypeError("Shape must be (u64, ...)")
    return Array(v_t)


@type_inferrer(P.reshape, nargs=2)
async def infer_type_reshape(track, v, shape):
    """Infer the return type of reshape."""
    shp_t = await shape['type']
    if (not isinstance(shp_t, Tuple) or
            not all(e == UInt(64) for e in shp_t.elements)):
        raise MyiaTypeError("Shape must be (u64, ...)")
    return await v['type']


@type_inferrer(P.dot, nargs=2)
async def infer_type_dot(track, a, b):
    """Infer the return type of dot."""
    t = await track.assert_same(a, b)
    return t


@type_inferrer(P.return_, nargs=1)
async def infer_type_return_(track, x):
    """Infer the return type of return_."""
    return await x['type']


@type_inferrer(P.list_map, nargs=2)
async def infer_type_list_map(track, f, xs):
    """Infer the return type of list_map."""
    f_t = await f['type']
    xs_t = await xs['type']
    if not isinstance(xs_t, List):
        raise MyiaTypeError('Expect list for list_map')
    xref = track.engine.vref(dict(type=xs_t.element_type))
    ret_t = await f_t(xref)
    return List(ret_t)


@type_inferrer(P.identity, nargs=1)
async def infer_type_identity(track, x):
    """Infer the return type of identity."""
    return await x['type']


@type_inferrer(P.resolve, nargs=2)
async def infer_type_resolve(track, data, item):
    """Infer the return type of resolve."""
    def chk(data_v, item_v):
        if not isinstance(data_v, Namespace):  # pragma: no cover
            raise MyiaTypeError('data argument to resolve must be Namespace.')
        if not isinstance(item_v, str):  # pragma: no cover
            raise MyiaTypeError('item argument to resolve must be string.')
    return await static_getter(track, data, item, (lambda x, y: x[y]), chk)


@type_inferrer(P.getattr, nargs=2)
async def infer_type_getattr(track, data, item):
    """Infer the return type of getattr."""
    def chk(data_v, item_v):
        if not isinstance(item_v, str):
            raise MyiaTypeError('item argument to getattr must be string.')
    return await static_getter(track, data, item, getattr, chk)


@type_inferrer(P.iter, nargs=1)
async def infer_type_iter(engine, xs):
    """Infer the return type of iter."""
    xs_t = await xs['type']
    if isinstance(xs_t, List):
        return Tuple([Int(64), xs_t])
    else:
        raise MyiaTypeError('Unsupported type for iter')


@type_inferrer(P.hasnext, nargs=1)
async def infer_type_hasnext(engine, it):
    """Infer the return type of hasnext."""
    it_t = await(it['type'])
    if isinstance(it_t, Tuple) \
            and len(it_t.elements) == 2 \
            and isinstance(it_t.elements[1], List):
        return Bool()
    else:  # pragma: no cover
        raise MyiaTypeError('Unsupported iterator type for hasnext')


@type_inferrer(P.next, nargs=1)
async def infer_type_next(engine, it):
    """Infer the return type of next."""
    it_t = await(it['type'])
    if isinstance(it_t, Tuple) \
            and len(it_t.elements) == 2 \
            and isinstance(it_t.elements[1], List):
        x_t = it_t.elements[1].element_type
        return Tuple([x_t, it_t])
    else:  # pragma: no cover
        raise MyiaTypeError('Unsupported iterator type for next')
