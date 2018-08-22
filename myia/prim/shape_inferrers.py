"""Definition of shape inference for primitives."""

import operator
from dataclasses import is_dataclass
from functools import partial, reduce

from ..infer import ANYTHING, GraphInferrer, register_inferrer, \
    PartialInferrer, Track, MyiaShapeError, Inferrer,  MetaGraphInferrer, \
    InferenceError
from ..ir import Graph, MetaGraph

from ..dtype import Array, Tuple, List, Class, TypeType, ismyiatype, \
    pytype_to_myiatype
from ..utils import Named

from . import ops as P
from .inferrer_utils import static_getter
from .ops import Primitive


def prod(iterable):
    """Return the product of the elements of the iterator."""
    return reduce(operator.mul, iterable, 1)


shape_inferrer_constructors = {}


NOSHAPE = Named('NOSHAPE')


class TupleShape:
    """Class to distinguish the shape of tuples items."""

    __slots__ = ['shape']

    def __init__(self, shape):
        """Create the shape."""
        self.shape = tuple(shape)

    def __repr__(self):
        return f"T{self.shape}"

    def __len__(self):
        return len(self.shape)

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.shape == other.shape)

    def __hash__(self):
        return hash((type(self), self.shape))


class ListShape:
    """Class to represent the shape of list elements."""

    __slots__ = ['shape']

    def __init__(self, shape):
        """Create the shape."""
        self.shape = shape

    def __repr__(self):
        return f"L{self.shape}"

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.shape == other.shape)

    def __hash__(self):
        return hash((type(self), self.shape))


class ClassShape:
    """Class to represent the shape of dataclass fields."""

    __slots__ = ['shape']

    def __init__(self, shape):
        """Create the shape."""
        self.shape = shape

    def __repr__(self):
        return f"C{self.shape}"

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.shape == other.shape)

    def __hash__(self):
        return hash((type(self), tuple(sorted(self.shape.items()))))


class ScalarShapeInferrer(Inferrer):
    """Shape inferrer for all primitives that don't take arrays."""

    def __init__(self, track):
        """Initialize the ScalarShapeInferrer."""
        super().__init__(track, 'scalar_shape_inferrer')

    async def __call__(self, *args):
        """Since no arrays are involved, there is no shape."""
        return NOSHAPE

    def provably_equivalent(self, other):
        """This is always equal to itself."""
        return type(self) == type(other)


def find_matching_shape(shps):
    """Returns a shape that matches all shapes in `shps`."""
    shp = shps[0]
    shps = shps[1:]

    if all(shp == s for s in shps):
        return shp

    if (not isinstance(shp, tuple) or
            any(not isinstance(s, tuple) for s in shps)):
        raise InferenceError("Mismatched element shapes in list")

    if not all(len(shp) == len(s) for s in shps):
        raise InferenceError("Arrays of differing ndim")

    shp = list(shp)
    for i, shp_i in enumerate(shp):
        if any(s[i] != shp_i for s in shps):
            shp[i] = ANYTHING

    return tuple(shp)


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
            if v in self.constructors:
                return self.constructors[v](self)
            else:
                return ScalarShapeInferrer(self)
        elif isinstance(v, Graph):
            return GraphInferrer(self, v, context)
        elif isinstance(v, MetaGraph):
            return MetaGraphInferrer(self, v)
        elif isinstance(v, tuple):
            return TupleShape(self.from_value(e, context) for e in v)
        elif isinstance(v, list):
            shps = [self.from_value(e, context) for e in v]
            if len(shps) == 0:
                return ListShape(NOSHAPE)  # pragma: no cover
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
        else:
            return getattr(v, 'shape', NOSHAPE)


shape_inferrer = partial(register_inferrer,
                         constructors=shape_inferrer_constructors)


@shape_inferrer(P.cons_tuple, nargs=2)
async def infer_shape_cons_tuple(track, head, tail):
    """Infer the shape for cons_tuple."""
    sht = await tail['shape']
    shh = await head['shape']
    return TupleShape((shh,) + sht.shape)


@shape_inferrer(P.head, nargs=1)
async def infer_shape_head(track, tup):
    """Infer the shape for head."""
    return (await tup['shape']).shape[0]


@shape_inferrer(P.tail, nargs=1)
async def infer_shape_tail(track, tup):
    """Infer the shape of tail."""
    return TupleShape((await tup['shape']).shape[1:])


@shape_inferrer(P.getitem, nargs=2)
async def infer_shape_getitem(track, seq, idx):
    """Infer the shape of getitem."""
    seq_t = await seq['type']

    if ismyiatype(seq_t, Tuple):
        seq_sh = await seq['shape']
        idx_v = await idx['value']
        assert idx_v is not ANYTHING
        return seq_sh.shape[idx_v]
    # For any other type
    raise InferenceError("Unknown type")  # pragma: no cover


@shape_inferrer(P.make_record, nargs=None)
async def infer_type_make_record(track, cls, *elems):
    """Infer the shape of make_record."""
    elem_shapes = [await x['shape'] for x in elems]
    cls_v = await cls['value']
    return ClassShape(dict(zip(cls_v.attributes.keys(), elem_shapes)))


@shape_inferrer(P.iter, nargs=1)
async def infer_shape_iter(track, seq):
    """Infer the shape of iter."""
    seq_sh = await seq['shape']
    return TupleShape(((), seq_sh))


@shape_inferrer(P.next, nargs=1)
async def infer_shape_next(track, it):
    """Infer the shape of next."""
    it_sh = await it['shape']
    it_t = await it['type']
    # This is probably wrong but it works for now
    data_sh = track.default({'type': it_t.elements[1]})
    return TupleShape((data_sh, it_sh))


@shape_inferrer(P.return_, nargs=1)
async def infer_shape_return(track, v):
    """Infer the shape of return."""
    return await v['shape']


@shape_inferrer(P.if_, nargs=3)
async def infer_shape_if(track, cond, tb, fb):
    """Infer the shape of if."""
    tb_inf = await tb['shape']
    fb_inf = await fb['shape']
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
        return await track.assert_same(tb_inf(), fb_inf(), refs=[tb, fb])
    else:
        raise AssertionError("Invalid condition value for if.")


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
        return await track.assert_same(tb, fb, refs=[tb, fb])
    else:
        raise AssertionError("Invalid condition value for switch.")


@shape_inferrer(P.partial, nargs=None)
async def infer_shape_partial(engine, fn, *args):
    """Infer the return type of partial."""
    fn_t = await fn['shape']
    return PartialInferrer(engine, fn_t, args)


@shape_inferrer(P.array_map, nargs=2)
async def infer_shape_array_map(track, fn, ary):
    """Infer the shape of array_map."""
    return await ary['shape']


@shape_inferrer(P.array_map2, nargs=3)
async def infer_shape_array_map2(track, fn, ary1, ary2):
    """Infer the shape of array_map2."""
    shp1 = await ary1['shape']
    shp2 = await ary2['shape']
    if len(shp1) != len(shp2):
        raise MyiaShapeError("Expect same shapes for array_map2")
    for a, b in zip(shp1, shp2):
        if a != b and a is not ANYTHING and b is not ANYTHING:
            raise MyiaShapeError("Expect same shapes for array_map2")
    return shp1


@shape_inferrer(P.list_map, nargs=2)
async def infer_shape_list_map(track, fn, lst):
    """Infer the shape of list_map."""
    shp = await lst['shape']
    typ = await lst['type']
    e = track.engine.vref({'type': typ.element_type, 'shape': shp.shape})
    return ListShape(await (await fn['shape'])(e))


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
    v_t = await v['type']
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


@shape_inferrer(P.dot, nargs=2)
async def infer_shape_dot(track, a, b):
    """Infer the shape of dot."""
    a_shp = await a['shape']
    b_shp = await b['shape']
    if len(a_shp) != 2 or len(b_shp) != 2:
        raise MyiaShapeError("dot needs matrix inputs")
    if (a_shp[1] != b_shp[0] and
        a_shp[1] is not ANYTHING and
            b_shp[0] is not ANYTHING):
        raise MyiaShapeError("Incompatible shapes in dot")
    return (a_shp[0], b_shp[1])


@shape_inferrer(P.resolve, nargs=2)
async def infer_shape_resolve(track, data, item):
    """Infer the shape of resolve."""
    return await static_getter(track, data, item, lambda x, y: x[y])


@shape_inferrer(P.getattr, nargs=2)
async def infer_shape_getattr(track, data, item):
    """Infer the shape of getattr."""
    data_typ = await data['type']
    if ismyiatype(data_typ, Class):
        item_v = await item['value']
        if item_v is ANYTHING:
            raise InferenceError(
                "getattr with non-constant item")  # pragma: no cover
        data_sh = await data['shape']
        return data_sh.shape[item_v]
    return await static_getter(track, data, item, getattr)


@shape_inferrer(P.identity, nargs=1)
async def infer_shape_identity(track, x):
    """Infer the shape of identity."""
    return await x['shape']


@shape_inferrer(P.scalar_to_array, nargs=1)
async def infer_shape_scalar_to_array(track, x):
    """Infer the shape of scalar_to_array."""
    return ()
