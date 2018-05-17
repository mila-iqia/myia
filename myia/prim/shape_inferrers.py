"""Definition of shape inference for primitives."""

from functools import partial

from ..infer import ANYTHING, GraphInferrer, register_inferrer, \
    Track, MyiaShapeError
from ..ir import Graph

from . import ops as P
from .ops import Primitive
from .py_implementations import shape


shape_inferrer_constructors = {}


class ShapeTrack(Track):
    """Infer the shape of a constant."""

    def __init__(self, engine, name, *,
                 constructors=shape_inferrer_constructors):
        """Intialize a ShapeTrack."""
        super().__init__(engine, name)
        self.constructors = constructors

    def from_value(self, v, context):
        """Infer the shape of a constant."""
        if isinstance(v, Primitive):
            return self.constructors[v](self.engine)
        elif isinstance(v, Graph):
            return GraphInferrer(self.engine, 'shape', v, context)
        else:
            return shape(v)


shape_inferrer = partial(register_inferrer,
                         constructors=shape_inferrer_constructors)


@shape_inferrer(P.map_array, nargs=3)
async def infer_shape_map_array(engine, fn, ary, ax):
    return await engine.get('shape', ary)


@shape_inferrer(P.scan_array, nargs=4)
async def infer_shape_scan_array(engine, fn, init, ary, ax):
    return await engine.get('shape', ary)


@shape_inferrer(P.reduce_array, nargs=4)
async def infer_shape_reduce_array(engine, fn, init, ary, ax):
    shp = await engine.get('shape', ary)
    ax_v = await engine.get('value', ax)
    if ax_v == ANYTHING:
        return (None,) * (len(shp) - 1)
    else:
        lshp = list(shp)
        del lshp[ax]
        return tuple(lshp)


@shape_inferrer(P.distribute, nargs=2)
async def infer_shape_distribute(engine, v, shape):
    shp = await engine.get('value', shape)
    if shp == ANYTHING:
        shp_t = await engine.get('type', shape)
        shp = (None,) * len(shp_t.elements)
    return shp


@shape_inferrer(P.dot, nargs=2)
async def infer_shape_dot(engine, a, b):
    a_shp = await engine.get('shape', a)
    b_shp = await engine.get('shape', b)
    if len(a_shp) != 2 or len(b_shp) != 2:
        raise MyiaShapeError("dot needs matrix inputs")
    if a[1] != b[0] and a[1] is not None and b[0] is not None:
        raise MyiaShapeError("Incompatible shapes in dot")
    return (a[0], b[1])
