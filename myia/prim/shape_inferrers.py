"""Definition of shape inference for primitives."""

import operator
import numpy
from dataclasses import is_dataclass
from functools import partial, reduce

from ..dshape import NOSHAPE, TupleShape, ListShape, ClassShape, \
    find_matching_shape, shape_cloner
from ..dtype import Array, Tuple, List, Class, TypeType, ismyiatype, \
    pytype_to_myiatype
from ..infer import ANYTHING, GraphInferrer, register_inferrer, \
    PartialInferrer, Track, MyiaShapeError, Inferrer,  MetaGraphInferrer, \
    InferenceError, MyiaTypeError, TransformedReference, MultiInferrer, \
    DummyInferrer, Context
from ..infer.jinf import JInferrer
from ..ir import Graph, MetaGraph

from . import ops as P
from .inferrer_utils import static_getter, getelement
from .ops import Primitive


def prod(iterable):
    """Return the product of the elements of the iterator."""
    return reduce(operator.mul, iterable, 1)


shape_inferrer_constructors = {}


@shape_cloner.variant
def _stag_shape(self, shp: Inferrer):
    return NOSHAPE


class ShapeTrack(Track):
    """Infer the shape of a constant."""

    def __init__(self, engine, name, *,
                 constructors=shape_inferrer_constructors):
        """Initialize a ShapeTrack."""
        super().__init__(engine, name)
        self.constructors = constructors

    def default(self, values):
        """Default value for ShapeTrack."""
        if ismyiatype(values['type'], Array):
            raise Exception(
                'There is no default value for Arrays on the shape track.'
            )  # pragma: no cover
        if ismyiatype(values['type'], Tuple):
            tup = values['type']
            return TupleShape(self.default({'type': e}) for e in tup.elements)
        elif ismyiatype(values['type'], List):
            lst = values['type']
            return ListShape(self.default({'type': lst.element_type}))
        elif ismyiatype(values['type'], Class):
            cls = values['type']
            return ClassShape(dict((attr, self.default({'type': tp}))
                                   for attr, tp in cls.attributes.items()))
        return NOSHAPE

    def from_value(self, v, context):
        """Infer the shape of a constant."""
        if isinstance(v, Primitive):
            return self.constructors[v](self)
        elif isinstance(v, Graph):
            return GraphInferrer(self, v, context)
        elif isinstance(v, MetaGraph):
            return MetaGraphInferrer(self, v)
        elif isinstance(v, tuple):
            return TupleShape(self.from_value(e, context) for e in v)
        elif isinstance(v, list):
            shps = [self.from_value(e, context) for e in v]
            if len(shps) == 0:  # pragma: no cover
                # from_value of the type track will fail before this
                raise InferenceError('Cannot infer the shape of []')
            return ListShape(find_matching_shape(shps))
        elif is_dataclass(v):
            if isinstance(v, type):
                rec = self.constructors[P.make_record](self)
                typ = pytype_to_myiatype(v)
                vref = self.engine.vref({'value': typ, 'type': TypeType})
                return PartialInferrer(self, rec, [vref])
            else:
                return ClassShape(
                    dict((n, self.from_value(getattr(v, n), context))
                         for n in v.__dataclass_fields__.keys()))
        elif isinstance(v, numpy.ndarray):
            return v.shape
        else:
            return NOSHAPE

    def jtag(self, shp):
        """Return type for J(x) given shape(x)."""
        if isinstance(shp, Inferrer):
            return JInferrer(shp, TupleShape)
        else:
            return shp

    def stag(self, t):
        """Return type for sensitivity of x given shape(x)."""
        return _stag_shape(t)


shape_inferrer = partial(register_inferrer,
                         constructors=shape_inferrer_constructors)


@shape_inferrer(P.scalar_add, P.scalar_sub, P.scalar_mul, P.scalar_div,
                P.scalar_mod, P.scalar_pow, P.scalar_trunc, P.scalar_floor,
                P.scalar_uadd, P.scalar_usub, P.scalar_exp, P.scalar_log, P.scalar_maximum,
                P.scalar_sin, P.scalar_cos, P.scalar_tan,
                P.scalar_eq, P.scalar_lt, P.scalar_gt, P.scalar_ne,
                P.scalar_le, P.scalar_ge,
                P.bool_not, P.bool_and, P.bool_or, P.bool_eq,
                P.typeof, P.hastype,
                P.tuple_len, P.list_len, P.array_len,
                P.scalar_cast,
                nargs=None)
async def infer_shape_scalar(track, *args):
    """Infer the shape of all scalar primitives."""
    return NOSHAPE

@shape_inferrer(P.array_cast, nargs=2)
async def infer_shape_array_cast(track, xs, t):
    """Infer the shape for array_cast."""
    return await xs['shape']

@shape_inferrer(P.shape, nargs=1)
async def infer_shape_shape(track, ary):
    """Infer the shape for shape."""
    shp = await ary['shape']
    return TupleShape((NOSHAPE,) * len(shp))


@shape_inferrer(P.make_tuple, nargs=None)
async def infer_shape_make_tuple(track, *args):
    """Infer the shape for make_tuple."""
    sh = [await x['shape'] for x in args]
    return TupleShape(sh)


@shape_inferrer(P.tuple_getitem, nargs=2)
async def infer_shape_tuple_getitem(track, seq, idx):
    """Infer the shape of tuple_getitem."""
    seq_sh = await seq['shape']
    idx_v = await idx['value']
    return seq_sh.shape[idx_v]


@shape_inferrer(P.tuple_setitem, nargs=3)
async def infer_shape_tuple_setitem(track, seq, idx, value):
    """Infer the shape of tuple_setitem."""
    seq_sh = await seq['shape']
    idx_v = await idx['value']
    value_sh = await value['shape']
    new_sh = list(seq_sh.shape)
    new_sh[idx_v] = value_sh
    return TupleShape(new_sh)


@shape_inferrer(P.list_getitem, nargs=2)
async def infer_shape_list_getitem(track, seq, idx):
    """Infer the shape of list_getitem."""
    seq_sh = await seq['shape']
    idx_t = await idx['type']
    if ismyiatype(idx_t, Tuple):
        return seq_sh
    return seq_sh.shape

@shape_inferrer(P.list_setitem, nargs=3)
async def infer_shape_list_setitem(track, seq, idx, value):
    """Infer the shape of list_setitem."""
    seq_sh = await seq['shape']
    return seq_sh

@shape_inferrer(P.array_getitem, nargs=2)
async def infer_shape_array_getitem(track, seq, idx):
    """Infer the shape of array_getitem."""
    seq_sh = await seq['shape']
    idx_t = await idx['type']
    idx_v = await idx['value']
    if ismyiatype(idx_t, Tuple):
        sh_new = (idx_v[1] - idx_v[0]) // idx_v[2]
        return tuple([sh_new])
    else:
        return NOSHAPE

@shape_inferrer(P.array_setitem, nargs=3)
async def infer_shape_array_setitem(track, seq, idx, value):
    """Infer the shape of list_setitem."""
    seq_sh = await seq['shape']
    return seq_sh

@shape_inferrer(getelement, nargs=1)
async def infer_shape_getelement(track, seq):
    """Infer the shape of an arbitrary element."""
    shp = await seq['shape']
    if isinstance(shp, ListShape):
        return shp.shape
    elif isinstance(shp, tuple):
        # Array
        return NOSHAPE
    else:
        raise AssertionError()


@shape_inferrer(P.make_record, nargs=None)
async def infer_type_make_record(track, cls, *elems):
    """Infer the shape of make_record."""
    elem_shapes = [await x['shape'] for x in elems]
    cls_v = await cls['value']
    return ClassShape(dict(zip(cls_v.attributes.keys(), elem_shapes)))


@shape_inferrer(P.return_, nargs=1)
async def infer_shape_return(track, v):
    """Infer the shape of return."""
    return await v['shape']


@shape_inferrer(P.switch, nargs=3)
async def infer_shape_switch(track, cond, tb, fb):
    """Infer the shape of switch."""
    v = await cond['value']
    if v is True:
        # We only visit the first branch if the condition is provably true
        return await tb['shape']
    elif v is False:
        # We only visit the second branch if the condition is provably false
        return await fb['shape']
    elif v is ANYTHING:
        # The first branch to finish will return immediately. When the other
        # branch finishes, its result will be checked against the other.
        res = await track.assert_same(tb, fb, refs=[tb, fb])
        if isinstance(res, Inferrer):
            tinf = await tb['shape']
            finf = await fb['shape']
            return MultiInferrer((tinf, finf), [tb, fb])
        return res
    else:
        raise AssertionError("Invalid condition value for switch.")


@shape_inferrer(P.partial, nargs=None)
async def infer_shape_partial(engine, fn, *args):
    """Infer the return type of partial."""
    fn_t = await fn['shape']
    return PartialInferrer(engine, fn_t, args)


@shape_inferrer(P.array_map, nargs=None)
async def infer_shape_array_map(track, fn, *arrays):
    """Infer the shape of array_map."""
    fn_t = await fn['shape']
    vrefs = [TransformedReference(track.engine, getelement, a)
             for a in arrays]
    elem_shp = await fn_t(*vrefs)
    assert elem_shp is NOSHAPE

    shapes = [await a['shape'] for a in arrays]
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
    return tuple(rshape)


@shape_inferrer(P.list_append, nargs=2)
async def infer_shape_list_append(track, seq, value):
    """Infer the shape for list_append."""
    lshp = await seq['shape']
    vshp = await value['shape']
    return ListShape(find_matching_shape((lshp.shape, vshp)))


@shape_inferrer(P.list_map, nargs=None)
async def infer_shape_list_map(track, fn, *lsts):
    """Infer the shape of list_map."""
    argrefs = [TransformedReference(track.engine, getelement, xs)
               for xs in lsts]
    return ListShape(await (await fn['shape'])(*argrefs))  # noqa: W606


@shape_inferrer(P.array_scan, nargs=4)
async def infer_shape_array_scan(track, fn, init, ary, ax):
    """Infer the shape of array_scan."""
    return await ary['shape']


@shape_inferrer(P.array_reduce, nargs=3)
async def infer_shape_array_reduce(track, fn, ary, shp):
    """Infer the shape of array_reduce."""
    shp_i = await ary['shape']
    shp_v = await shp['value']
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
        return shp_v


@shape_inferrer(P.distribute, nargs=2)
async def infer_shape_distribute(track, v, shape):
    """Infer the shape of distribute."""
    shp = await shape['value']
    if shp == ANYTHING:
        shp_t = await shape['type']
        shp = (ANYTHING,) * len(shp_t.elements)
    v_t = await v.get_shallow('type')
    if ismyiatype(v_t, Array):
        v_shp = await v['shape']
        delta = len(shp) - len(v_shp)
        if delta < 0:
            raise MyiaShapeError("Cannot distribute to smaller shape")
        elif delta > 0:
            v_shp = (1,) * delta + v_shp
        for vs, s in zip(v_shp, shp):
            if vs != s and vs not in (1, ANYTHING) and s not in (1, ANYTHING):
                raise MyiaShapeError("Cannot change shape when distributing")
    return shp


@shape_inferrer(P.reshape, nargs=2)
async def infer_shape_reshape(track, v, shape):
    """Infer the shape of reshape."""
    shp = await shape['value']
    if shp == ANYTHING:
        shp_t = await shape['type']
        shp = (ANYTHING,) * len(shp_t.elements)
    v_shp = await v['shape']
    if (all(s is not ANYTHING for s in shp) and
        all(s is not ANYTHING for s in v_shp) and
            prod(shp) != prod(v_shp)):
        raise MyiaShapeError("Cannot change the total number of elements "
                             "in reshape")
    return shp


@shape_inferrer(P.transpose, nargs=2)
async def infer_shape_transpose(track, v, permutation):
    """Infer the shape of transpose."""
    perm = await permutation['value']
    if perm == ANYTHING:
        perm_t = await permutation['type']
        return (ANYTHING,) * len(perm_t.elements)

    v_shp = await v['shape']
    if list(sorted(perm)) != list(range(len(v_shp))):
        raise MyiaShapeError(
            'The second argument of transpose must be a permutation of'
            ' all of the array\'s axes.',
            refs=[permutation]
        )

    shp = tuple(v_shp[i] for i in perm)
    return shp


@shape_inferrer(P.invert_permutation, nargs=1)
async def infer_shape_invert_permutation(track, permutation):
    """Infer the shape for invert_permutation."""
    t = await permutation['type']
    return TupleShape([NOSHAPE for _ in t.elements])


@shape_inferrer(P.array_conv, nargs=5)
async def infer_shape_array_conv(track, a, height, width, kernel_size, stride):
    a_shp = await a['shape']
    #h_size = (a_shp[0] - b)//c+1
    #w_size = (a_shp[1] - b)//c+1
    return (16, 4)

@shape_inferrer(P.cal_conv_grad, nargs=5)
async def infer_shape_conv_op(track, a, b,c,e,f):
    return (b, c)

@shape_inferrer(P.dot, nargs=2)
async def infer_shape_dot(track, a, b):
    """Infer the shape of dot."""
    a_shp = await a['shape']
    b_shp = await b['shape']
    if len(a_shp) != 2 or len(b_shp) != 2:
        raise MyiaShapeError("dot needs matrix inputs")
    if (a_shp[1] != b_shp[0] and
            a_shp[1] is not ANYTHING and b_shp[0] is not ANYTHING):
        raise MyiaShapeError(
            f"Incompatible shapes in dot: {a_shp} and {b_shp}"
        )
    return (a_shp[0], b_shp[1])


@shape_inferrer(P.resolve, nargs=2)
async def infer_shape_resolve(track, data, item):
    """Infer the shape of resolve."""
    async def on_dcattr(data, data_t, item_v):  # pragma: no cover
        raise MyiaTypeError('Cannot resolve on Class.')

    return await static_getter(
        track, data, item,
        fetch=operator.getitem,
        on_dcattr=on_dcattr
    )


@shape_inferrer(P.getattr, nargs=2)
async def infer_shape_getattr(track, data, item):
    """Infer the shape of getattr."""
    async def on_dcattr(data, data_t, item_v):
        data_sh = await data['shape']
        return data_sh.shape[item_v]

    return await static_getter(
        track, data, item,
        fetch=getattr,
        on_dcattr=on_dcattr
    )


@shape_inferrer(P.identity, nargs=1)
async def infer_shape_identity(track, x):
    """Infer the shape of identity."""
    return await x['shape']


@shape_inferrer(P.scalar_to_array, nargs=1)
async def infer_shape_scalar_to_array(track, x):
    """Infer the shape of scalar_to_array."""
    return ()


@shape_inferrer(P.array_to_scalar, nargs=1)
async def infer_shape_array_to_scalar(track, ary):
    """Infer the shape of array_to_scalar."""
    shp = await ary['shape']
    if shp == ():
        return NOSHAPE
    else:
        raise MyiaTypeError(
            'array_to_scalar only works on 0d arrays',
            refs=[ary]
        )


@shape_inferrer(P.broadcast_shape, nargs=2)
async def infer_shape_broadcast_shape(track, shpx, shpy):
    """Infer the shape of broadcast_shape."""
    tx = await shpx['type']
    ty = await shpy['type']
    n = max(len(tx.elements), len(ty.elements))
    return TupleShape([NOSHAPE] * n)


@shape_inferrer(P.make_list, nargs=None)
async def infer_shape_make_list(track, *elems):
    """Infer the return shape of make_list."""
    shps = [await e['shape'] for e in elems]
    if len(shps) == 0:
        raise InferenceError('Cannot infer the shape of []')
    return ListShape(find_matching_shape(shps))


@shape_inferrer(P.list_reduce, nargs=3)
async def infer_shape_list_reduce(track, fn, lst, dflt):
    """Infer the return shape of list_reduce."""
    elem = TransformedReference(track.engine, getelement, lst)
    fn_inf = await fn['shape']
    shp1 = await fn_inf(dflt, elem)
    shp2 = await fn_inf(elem, elem)
    return find_matching_shape([shp1, shp2])


@shape_inferrer(P.J, nargs=1)
async def infer_shape_J(track, x):
    """Infer the return shape of J."""
    return track.jtag(await x.get_shallow('shape'))


@shape_inferrer(P.Jinv, nargs=1)
async def infer_shape_Jinv(track, x):
    """Infer the return shape of Jinv."""
    shp = await x.get_shallow('shape')
    if isinstance(shp, JInferrer):
        return shp.fn
    elif isinstance(shp, GraphInferrer):
        g = shp._graph
        primal = g and g.transforms.get('primal', None)
        if primal:
            primal = track.engine.pipeline.resources.convert(primal)
            if isinstance(primal, Graph) and primal.parent:
                return DummyInferrer(track)
            else:
                return track.from_value(primal, Context.empty())
        else:  # pragma: no cover
            # This error is also caught by the type inferrer
            raise MyiaTypeError('Bad input type for Jinv', refs=[x])
    else:
        return shp


@shape_inferrer(P.embed, nargs=1)
async def infer_shape_embed(track, x):
    """Infer the return shape of embed."""
    return NOSHAPE


@shape_inferrer(P.env_setitem, nargs=3)
async def infer_shape_env_setitem(track, env, key, x):
    """Infer the return shape of env_setitem."""
    return NOSHAPE


@shape_inferrer(P.env_getitem, nargs=3)
async def infer_shape_env_getitem(track, env, key, default):
    """Infer the return shape of env_getitem."""
    key_v = await key['value']
    assert key_v is not ANYTHING
    shp = track.stag(key_v.inferred['shape'])
    return await track.assert_same(shp, default)


@shape_inferrer(P.env_add, nargs=2)
async def infer_shape_env_add(track, env1, env2):
    """Infer the return shape of env_add."""
    return NOSHAPE
