"""Definition of shape inference for primitives."""

import operator
from functools import partial, reduce

from ..infer import ANYTHING, GraphInferrer, register_inferrer, \
    PartialInferrer, Track, MyiaShapeError, Inferrer,  MetaGraphInferrer
from ..ir import Graph, MetaGraph
from ..dtype import Array, ismyiatype
from ..utils import is_dataclass_type

from . import ops as P
from .inferrer_utils import static_getter
from .ops import Primitive


def prod(iterable):
    """Return the product of the elements of the iterator."""
    return reduce(operator.mul, iterable, 1)


shape_inferrer_constructors = {}


class ScalarShapeInferrer(Inferrer):
    """Shape inferrer for all primitives that don't take arrays."""

    def __init__(self, track):
        """Initialize the ScalarShapeInferrer."""
        super().__init__(track, 'scalar_shape_inferrer')

    async def __call__(self, *args):
        """Since no arrays are involved, the shape is always ()."""
        return ()

    def provably_equivalent(self, other):
        """This is always equal to itself."""
        return type(self) == type(other)


class ShapeTrack(Track):
    """Infer the shape of a constant."""

    def __init__(self, engine, name, *,
                 constructors=shape_inferrer_constructors):
        """Initialize a ShapeTrack."""
        super().__init__(engine, name)
        self.constructors = constructors

    def default(self, values):
        """Default value for ShapeTrack."""
        if isinstance(values['type'], Array):
            raise Exception(
                'There is no default value for Arrays on the shape track.'
            )  # pragma: no cover
        return ()

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
        elif is_dataclass_type(v):
            raise NotImplementedError('Dataclasses are not supported yet')
        else:
            return getattr(v, 'shape', ())


shape_inferrer = partial(register_inferrer,
                         constructors=shape_inferrer_constructors)


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
    return await static_getter(track, data, item, getattr)


@shape_inferrer(P.identity, nargs=1)
async def infer_shape_identity(track, x):
    """Infer the shape of identity."""
    return await x['shape']


@shape_inferrer(P.scalar_to_array, nargs=1)
async def infer_shape_scalar_to_array(track, x):
    """Infer the shape of scalar_to_array."""
    return ()
