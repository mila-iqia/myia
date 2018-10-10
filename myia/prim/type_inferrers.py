"""Definitions of type inference for primitives."""


from functools import partial
from operator import getitem

from ..dtype import Int, Float, Bool, Tuple, List, Array, UInt, Number, \
    TypeType, Class, Function, pytype_to_myiatype, Problem, type_cloner
from ..infer import ANYTHING, GraphInferrer, PartialInferrer, \
    MyiaTypeError, register_inferrer, Track, MetaGraphInferrer, \
    ExplicitInferrer, VOID, TransformedReference
from ..ir import Graph, MetaGraph
from ..utils import Namespace, Var, RestrictedVar, is_dataclass_type

from ..dtype import ismyiatype
from . import ops as P
from .inferrer_utils import static_getter, getelement
from .ops import Primitive
from .py_implementations import typeof, issubtype


type_inferrer_constructors = {}


_number_types = [
    Int[8], Int[16], Int[32], Int[64],
    UInt[8], UInt[16], UInt[32], UInt[64],
    Float[16], Float[32], Float[64],
]


def _shape_type(t):  # noqa: D400
    """Tuple[UInt[64] ...]"""
    # The docstring is used in the error message.
    return ismyiatype(t, Tuple) and all(x == UInt[64] for x in t.elements)


@type_cloner.variant
def _import_type(self, t: Function, track):
    return ExplicitInferrer(
        track,
        [self(t2, track) for t2 in t.arguments],
        self(t.retval, track)
    )


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

    async def infer_constant(self, ctref):
        """Get the property for a ref of a Constant node."""
        v = self.engine.pipeline.resources.convert(ctref.node.value)
        t = self.from_value(v, ctref.context)
        if ismyiatype(t, Number):
            v = RestrictedVar(_number_types)
            prio = 1 if ismyiatype(t, Float) else 0
            return self.engine.loop.create_var(v, t, prio)
        else:
            return t

    def from_value(self, v, context):
        """Infer the type of a constant."""
        if isinstance(v, Primitive):
            return self.constructors[v](self)
        elif isinstance(v, Graph):
            return GraphInferrer(self, v, context)
        elif isinstance(v, MetaGraph):
            return MetaGraphInferrer(self, v)
        elif is_dataclass_type(v):
            rec = self.constructors[P.make_record](self)
            typ = pytype_to_myiatype(v)
            vref = self.engine.vref({'value': typ, 'type': TypeType})
            return PartialInferrer(self, rec, [vref])
        else:
            return typeof(v)

    def from_external(self, t):
        """Convert a type provided outside the inferrer.

        This will replace every Function type by an ExplicitInferrer.
        """
        return _import_type(t, self)

    def default(self, values):
        """Return a default type; this method raises an exception."""
        raise Exception('There is no default value for the type track.') \
            # pragma: no cover


########################
# Default constructors #
########################


type_inferrer = partial(register_inferrer,
                        constructors=type_inferrer_constructors)


@type_inferrer(P.switch, nargs=3)
async def infer_type_switch(track, cond, tb, fb):
    """Infer the return type of switch."""
    await track.check(Bool, cond)
    v = await cond['value']
    if v is True:
        # We only visit the first branch if the condition is provably true
        return await tb['type']
    elif v is False:
        # We only visit the second branch if the condition is provably false
        return await fb['type']
    elif v is ANYTHING:
        # The first branch to finish will return immediately. When the other
        # branch finishes, its result will be checked against the other.
        return await track.assert_same(tb, fb, refs=[tb, fb])
    else:
        raise AssertionError("Invalid condition value for switch")


@type_inferrer(P.partial, nargs=None)
async def infer_type_partial(track, fn, *args):
    """Infer the return type of partial."""
    fn_t = await fn['type']
    return PartialInferrer(track, fn_t, args)


@type_inferrer(P.scalar_cast, nargs=2)
async def infer_type_scalar_cast(track, x, t):
    """Infer the return type of scalar_cast."""
    await track.will_check(Number, x)
    await track.check(TypeType, t)
    new_t = await t['value']
    if new_t is ANYTHING:
        raise MyiaTypeError(f'Type to cast to must be known at compile time.')
    elif not ismyiatype(new_t, Number):
        raise MyiaTypeError(f'Cannot cast to {new_t}')
    return new_t


@type_inferrer(P.make_tuple, nargs=None)
async def infer_type_make_tuple(track, *args):
    """Infer the return type of make_tuple."""
    elts = [await x['type'] for x in args]
    return Tuple[elts]


@type_inferrer(P.tail, nargs=1)
async def infer_type_tail(track, tup):
    """Infer the return type of tail."""
    tup_t = await track.check(Tuple, tup)
    if not len(tup_t.elements) >= 1:
        raise MyiaTypeError('tail on empty tuple', refs=[tup])
    return Tuple[tup_t.elements[1:]]


@type_inferrer(P.tuple_getitem, nargs=2)
async def infer_type_tuple_getitem(track, seq, idx):
    """Infer the return type of tuple_getitem."""
    seq_t = await track.check(Tuple, seq)
    await track.check(Int, idx)
    idx_v = await idx['value']
    if idx_v is ANYTHING:
        raise MyiaTypeError('Tuples must be indexed with a constant')
    return seq_t.elements[idx_v]


@type_inferrer(P.list_getitem, nargs=2)
async def infer_type_list_getitem(track, seq, idx):
    """Infer the return type of list_getitem."""
    seq_t = await track.check(List, seq)
    await track.check(Int, idx)
    return seq_t.element_type


@type_inferrer(getelement, nargs=1)
async def infer_type_getelement(track, seq):
    """Infer the return type of getting some arbitrary element."""
    seq_t = await track.check((List, Array), seq)
    if ismyiatype(seq_t, List):
        return seq_t.element_type
    else:
        return seq_t.elements


@type_inferrer(P.tuple_setitem, nargs=3)
async def infer_type_tuple_setitem(track, seq, idx, value):
    """Infer the return type of tuple_setitem."""
    seq_t = await track.check(Tuple, seq)
    await track.check(Int, idx)
    idx_v = await idx['value']
    if idx_v is ANYTHING:
        raise MyiaTypeError('Tuples must be indexed with a constant')
    value_t = await value['type']
    elts = seq_t.elements
    try:
        elts[idx_v]
    except IndexError:
        raise MyiaTypeError('Index out of bounds')
    new_elts = tuple([*elts[:idx_v], value_t, *seq_t.elements[idx_v + 1:]])
    return Tuple[new_elts]


@type_inferrer(P.list_setitem, nargs=3)
async def infer_type_list_setitem(track, seq, idx, value):
    """Infer the return type of list_setitem."""
    seq_t = await track.check(List, seq)
    await track.check(Int, idx)
    await track.will_check(seq_t.element_type, value)
    return seq_t


@type_inferrer(P.typeof, nargs=1)
async def infer_type_typeof(track, _):
    """Infer the return type of typeof."""
    return TypeType


@type_inferrer(P.hastype, nargs=2)
async def infer_type_hastype(track, x, t):
    """Infer the return type of hastype."""
    def ismyiatype(x):  # noqa: D400
        """Type"""
        return x is TypeType

    await track.check(ismyiatype, t)
    return Bool


@type_inferrer(P.bool_not, nargs=1)
async def infer_type_bool_not(track, x):
    """Infer the return type of not."""
    await track.will_check(Bool, x)
    return Bool


@type_inferrer(P.bool_and, nargs=2)
async def infer_type_bool_and(track, x, y):
    """Infer the return type of bool_and."""
    await track.will_check(Bool, x, y)
    return Bool


@type_inferrer(P.bool_or, nargs=2)
async def infer_type_bool_or(track, x, y):
    """Infer the return type of bool_or."""
    await track.will_check(Bool, x, y)
    return Bool


@type_inferrer(P.scalar_eq, P.scalar_ne, nargs=2)
async def infer_type_generic_compare(track, x, y):
    """Infer the return type of a generic comparison operator."""
    await track.will_check(Number, x, y)
    return Bool


@type_inferrer(P.scalar_lt, P.scalar_gt, P.scalar_le, P.scalar_ge, nargs=2)
async def infer_type_arith_compare(track, x, y):
    """Infer the return type of an arithmetic comparison operator."""
    await track.will_check(Number, x, y)
    return Bool


@type_inferrer(P.scalar_uadd, P.scalar_usub, P.scalar_floor, P.scalar_trunc,
               nargs=1)
async def infer_type_arith_unary(track, x):
    """Infer the return type of a unary arithmetic operator."""
    return await track.will_check(Number, x)


@type_inferrer(P.scalar_exp, P.scalar_log, P.scalar_sin,
               P.scalar_cos, P.scalar_tan, nargs=1)
async def infer_type_arith_unary_float(track, x):
    """Infer the return type of a floating point unary arithmetic operator."""
    return await track.will_check(Float, x)


@type_inferrer(P.scalar_add, P.scalar_sub, P.scalar_mul, P.scalar_div,
               P.scalar_mod, P.scalar_pow, nargs=2)
async def infer_type_arith_bin(track, x, y):
    """Infer the return type of a binary arithmetic operator."""
    return await track.will_check(Number, x, y)


@type_inferrer(P.shape, nargs=1)
async def infer_type_shape(track, ary):
    """Infer the return type of shape."""
    await track.check(Array, ary)
    shp = await ary['shape']
    return Tuple[[UInt[64]]*len(shp)]


@type_inferrer(P.array_map, nargs=None)
async def infer_type_array_map(track, fn, *arrays):
    """Infer the return type of array_map."""
    fn_t = await fn['type']
    await track.check(Array, *arrays)
    vrefs = [TransformedReference(track.engine, getelement, a)
             for a in arrays]
    return Array[await fn_t(*vrefs)]


@type_inferrer(P.array_reduce, nargs=3)
async def infer_type_reduce(track, fn, ary, shp):
    """Infer the return type of array_reduce."""
    fn_t = await fn['type']
    await track.check(_shape_type, shp)
    await track.check(Array, ary)
    xref = TransformedReference(track.engine, getelement, ary)
    res_elem_t = await fn_t(xref, xref)
    return Array[res_elem_t]


@type_inferrer(P.array_scan, nargs=4)
async def infer_type_across_array(track, fn, init, ary, ax):
    """Infer the return type of scan/array_reduce."""
    fn_t = await fn['type']
    init_t = await init['type']
    ax_t = await ax['type']
    ary_t = await track.check(Array, ary)
    if not ary_t.elements == init_t:
        raise MyiaTypeError("Initial value must have the same type "
                            "as array elements")
    if not ax_t == UInt[64]:
        raise MyiaTypeError("Axis must be u64")
    xref = TransformedReference(track.engine, getelement, ary)
    return Array[await fn_t(xref, xref)]


@type_inferrer(P.distribute, nargs=2)
async def infer_type_distribute(track, v, shp):
    """Infer the return type of distribute."""
    v_t = await track.check(Array, v)
    await track.check(_shape_type, shp)
    return v_t


@type_inferrer(P.reshape, nargs=2)
async def infer_type_reshape(track, v, shape):
    """Infer the return type of reshape."""
    v_t = await track.check(Array, v)
    await track.check(_shape_type, shape)
    return v_t


@type_inferrer(P.dot, nargs=2)
async def infer_type_dot(track, a, b):
    """Infer the return type of dot."""
    return await track.will_check(Array, a, b)


@type_inferrer(P.return_, nargs=1)
async def infer_type_return_(track, x):
    """Infer the return type of return_."""
    return await x.get_raw('type')


@type_inferrer(P.list_map, nargs=None)
async def infer_type_list_map(track, f, *lsts):
    """Infer the return type of list_map."""
    f_t = await f['type']
    await track.check(List, *lsts)
    argrefs = [TransformedReference(track.engine, getelement, xs)
               for xs in lsts]
    ret_t = await f_t(*argrefs)
    return List[ret_t]


@type_inferrer(P.identity, nargs=1)
async def infer_type_identity(track, x):
    """Infer the return type of identity."""
    return await x.get_raw('type')


@type_inferrer(P.resolve, nargs=2)
async def infer_type_resolve(track, data, item):
    """Infer the return type of resolve."""
    def chk(data_v, item_v):
        if not isinstance(data_v, Namespace):  # pragma: no cover
            raise MyiaTypeError(
                f'data argument to resolve must be Namespace, not {data_v}'
            )
        if not isinstance(item_v, str):  # pragma: no cover
            raise MyiaTypeError(
                f'item argument to resolve must be a string, not {item_v}.'
            )

    async def on_dcattr(data, data_t, item_v):  # pragma: no cover
        raise MyiaTypeError('Cannot resolve on Class.')

    return await static_getter(
        track, data, item,
        fetch=getitem,
        on_dcattr=on_dcattr,
        chk=chk
    )


@type_inferrer(P.getattr, nargs=2)
async def infer_type_getattr(track, data, item):
    """Infer the return type of getattr."""
    def chk(data_v, item_v):
        if not isinstance(item_v, str):
            raise MyiaTypeError(
                f'item argument to getattr must be string, not {item_v}.'
            )

    async def on_dcattr(data, data_t, item_v):
        return data_t.attributes[item_v]

    return await static_getter(
        track, data, item,
        fetch=getattr,
        on_dcattr=on_dcattr,
        chk=chk
    )


@type_inferrer(P.scalar_to_array, nargs=1)
async def infer_type_scalar_to_array(track, x):
    """Infer the return type of scalar_to_array."""
    x_t = await track.will_check(Number, x)
    return Array[x_t]


@type_inferrer(P.broadcast_shape, nargs=2)
async def infer_type_broadcast_shape(track, xs, ys):
    """Infer the return type of broadcast_shape."""
    xs_t, ys_t = await track.check(_shape_type, xs, ys)
    shp_xs_n = len(xs_t.elements)
    shp_ys_n = len(ys_t.elements)
    return Tuple[[UInt[64] for i in range(max(shp_xs_n, shp_ys_n))]]


@type_inferrer(P.make_record, nargs=None)
async def infer_type_make_record(track, cls, *elems):
    """Infer the return type of make_record."""
    elem_types = [await x['type'] for x in elems]
    cls_v = await cls['value']

    ret_t = Class[
        cls_v.tag,
        dict(zip(cls_v.attributes.keys(), elem_types)),
        cls_v.methods
    ]

    if not issubtype(ret_t, cls_v):
        raise MyiaTypeError(
            f'Constructor {cls_v} cannot be called'
            f' with argument types {tuple(elem_types)}',
            refs=elems,
        )

    return ret_t


@type_inferrer(P.tuple_len, nargs=1)
async def infer_type_tuple_len(track, xs):
    """Infer the return type of tuple_len."""
    await track.will_check(Tuple, xs)
    return Int[64]


@type_inferrer(P.list_len, nargs=1)
async def infer_type_list_len(track, xs):
    """Infer the return type of list_len."""
    await track.will_check(List, xs)
    return Int[64]


@type_inferrer(P.array_len, nargs=1)
async def infer_type_array_len(track, xs):
    """Infer the return type of array_len."""
    await track.will_check(Array, xs)
    return Int[64]


@type_inferrer(P.make_list, nargs=None)
async def infer_type_make_list(track, *elems):
    """Infer the return type of make_list."""
    if len(elems) == 0:
        v = Var('empty')
        t = track.engine.loop.create_var(v, Problem[VOID], -1000)
    else:
        t = await track.assert_same(*elems, refs=elems)
    return List[t]


@type_inferrer(P.list_reduce, nargs=3)
async def infer_type_list_reduce(track, fn, lst, dflt):
    """Infer the return type of list_reduce."""
    fn_t = await fn['type']
    await track.check(List, lst)
    xref = TransformedReference(track.engine, getelement, lst)
    res_elem_t = await track.assert_same(fn_t(xref, xref), dflt)
    return res_elem_t
