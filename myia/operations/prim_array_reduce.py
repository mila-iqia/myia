"""Definitions for the primitive `array_reduce`."""

import numpy as np

from ..lib import (
    ANYTHING,
    SHAPE,
    TYPE,
    AbstractArray,
    AbstractFunctionBase,
    MetaGraph,
    MyiaShapeError,
    bprop_to_grad_transform,
    build_value,
    force_pending,
    newenv,
    standard_prim,
    u64tup_typecheck,
)
from ..operations import distribute, shape, zeros_like
from . import primitives as P


def pyimpl_array_reduce(fn, array, shp):
    """Implement `array_reduce`."""
    idtype = array.dtype
    ufn = np.frompyfunc(fn, 2, 1)
    delta = len(array.shape) - len(shp)
    if delta < 0:
        raise ValueError("Shape to reduce to cannot be larger than original")

    def is_reduction(ishp, tshp):
        if tshp == 1 and ishp > 1:
            return True
        elif tshp != ishp:
            raise ValueError("Dimension mismatch for reduce")
        else:
            return False

    reduction = [
        (delta + idx if is_reduction(ishp, tshp) else None, True)
        for idx, (ishp, tshp) in enumerate(zip(array.shape[delta:], shp))
    ]

    reduction = [(i, False) for i in range(delta)] + reduction

    for idx, keep in reversed(reduction):
        if idx is not None:
            array = ufn.reduce(array, axis=idx, keepdims=keep)

    if not isinstance(array, np.ndarray):
        # Force result to be ndarray, even if it's 0d
        array = np.array(array)

    array = array.astype(idtype)

    return array


def debugvm_array_reduce(vm, fn, array, shp):
    """Implement `array_reduce` for the debug VM."""

    def fn_(a, b):
        return vm.call(fn, [a, b])

    return pyimpl_array_reduce(fn_, array, shp)


@standard_prim(P.array_reduce)
async def infer_array_reduce(
    self,
    engine,
    fn: AbstractFunctionBase,
    a: AbstractArray,
    shp: u64tup_typecheck,
):
    """Infer the return type of primitive `array_reduce`."""
    shp_i = await force_pending(a.xshape())
    shp_v = build_value(shp, default=ANYTHING)
    if shp_v == ANYTHING:
        raise AssertionError(
            "We currently require knowing the shape for reduce."
        )
        # return (ANYTHING,) * (len(shp_i) - 1)
    else:
        delta = len(shp_i) - len(shp_v)
        if delta < 0 or any(
            1 != s1 != ANYTHING and 1 != s2 != ANYTHING and s1 != s2
            for s1, s2 in zip(shp_i[delta:], shp_v)
        ):
            raise MyiaShapeError(
                f"Incompatible dims for reduce: {shp_i}, {shp_v}"
            )

    res = await engine.execute(fn, a.element, a.element)
    return type(a)(res, {SHAPE: shp_v, TYPE: a.xtype()})


def bprop_sum(fn, xs, shp, out, dout):  # pragma: no cover
    """Backpropagator for sum(xs) = array_reduce(scalar_add, xs, shp)."""
    return (newenv, distribute(dout, shape(xs)), zeros_like(shp))


class ArrayReduceGradient(MetaGraph):
    """Generate the gradient graph for array_reduce.

    For the time being, the gradient of array_reduce is only supported
    over the `scalar_add` operation (sum, basically).
    """

    def generate_graph(self, args):
        """Generate the gradient graph."""
        # BUG: We only support gradients for sum (scalar_add as the reduction
        # function). However, it is currently not possible to check what the
        # reduction function is, due to erasure of the information when
        # GraphFunction and so on are converted to VirtualFunction.
        return bprop_to_grad_transform(P.array_reduce)(bprop_sum)


__operation_defaults__ = {
    "name": "array_reduce",
    "registered_name": "array_reduce",
    "mapping": P.array_reduce,
    "python_implementation": pyimpl_array_reduce,
}


__primitive_defaults__ = {
    "name": "array_reduce",
    "registered_name": "array_reduce",
    "type": "backend",
    "python_implementation": pyimpl_array_reduce,
    "debugvm_implementation": debugvm_array_reduce,
    "inferrer_constructor": infer_array_reduce,
    "grad_transform": ArrayReduceGradient(name="array_reduce_gradient"),
}
