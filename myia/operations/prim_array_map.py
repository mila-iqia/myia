"""Definitions for the primitive `array_map`."""

import numpy as np

from ..debug.label import short_relation_symbols as syms
from ..grad import default_grad_flags
from ..lib import (
    ANYTHING,
    SHAPE,
    TYPE,
    AbstractArray,
    AbstractFunction,
    Graph,
    MetaGraph,
    MyiaShapeError,
    MyiaTypeError,
    newenv,
    standard_prim,
)
from . import primitives as P


def pyimpl_array_map(fn, *arrays):
    """Implement `array_map`."""
    return np.vectorize(fn)(*arrays)


def debugvm_array_map(vm, fn, *arrays):
    """Implement `array_map` for the debug VM."""
    def fn_(*args):
        return vm.call(fn, args)
    return pyimpl_array_map(fn_, *arrays)


@standard_prim(P.array_map)
async def infer_array_map(self, engine, fn: AbstractFunction, *arrays):
    """Infer the return type of primitive `array_map`."""
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


class ArrayMapGradient(MetaGraph):
    """Generate the gradient graph for array_map.

    Sketch of the transform:

        array_map(f, xs, ys, ...) =>

        def fprop_array_map(jf, jxs, jys, ...):
            f, xs, ys, ... = Jinv(jf), Jinv(jxs), Jinv(jys), ...
            ret = array_map(f, xs, ys, ...)

            def bprop_array_map(dout):
                df = newenv
                f_dxs = lambda d, jx, jy, ...: jf(jx, jy, ...)[1](d)[1]
                dxs = array_map(f_dxs, dout, jxs, jys, ...)
                f_dys = lambda d, jx, jy, ...: jf(jx, jy, ...)[1](d)[2]
                dys = array_map(f_dys, dout, jxs, jys, ...)
                ...
                return df, dxs, dys, ...

            return ret, bprop_array_map
    """

    def generate_graph(self, absargs):
        """Generate the gradient graph."""
        g = Graph()
        nargs = len(absargs) - 1
        params = [g.add_parameter() for _ in range(nargs + 1)]
        jf, *jargs = params
        f, *args = [g.apply(P.Jinv, p) for p in params]
        ret = g.apply(P.array_map, f, *args)

        b = Graph()
        dout = b.add_parameter()

        results = []

        for i in range(nargs):
            func = Graph()
            fparams = [func.add_parameter() for _ in range(nargs + 1)]
            fparams[0].debug.name = f'{syms["grad_sens"]}out'
            fjparams = [func.apply(P.J, p) for p in fparams]
            call = func.apply(jf, *fjparams[1:])
            bprop = func.apply(P.tuple_getitem, call, 1)
            sens = func.apply(bprop, fparams[0])
            func.output = func.apply(P.tuple_getitem, sens, i + 1)
            result = b.apply(P.array_map, func, dout, *args)
            results.append(result)

        b.output = b.apply(P.make_tuple, newenv, newenv, *results)

        ret = g.apply(P.J, ret)
        g.output = g.apply(P.make_tuple, ret, b)

        b.flags.update(default_grad_flags)
        g.flags.update(default_grad_flags)

        return g


__operation_defaults__ = {
    'name': 'array_map',
    'registered_name': 'array_map',
    'mapping': P.array_map,
    'python_implementation': pyimpl_array_map,
}


__primitive_defaults__ = {
    'name': 'array_map',
    'registered_name': 'array_map',
    'type': 'backend',
    'python_implementation': pyimpl_array_map,
    'debugvm_implementation': debugvm_array_map,
    'inferrer_constructor': infer_array_map,
    'grad_transform': ArrayMapGradient(name='array_map_gradient'),
}
