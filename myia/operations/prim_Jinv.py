"""Definitions for the primitive `Jinv`."""

from ..lib import (
    AbstractFunction,
    AbstractJTagged,
    Context,
    DummyFunction,
    Graph,
    GraphFunction,
    JTransformedFunction,
    MyiaTypeError,
    Primitive,
    PrimitiveFunction,
    VirtualFunction,
    bprop_to_grad_transform,
    standard_prim,
)
from ..operations import J
from ..utils.errors import untested_legacy
from . import primitives as P


@standard_prim(P.Jinv)
async def infer_Jinv(self, engine, x):
    """Infer the return type of primitive `Jinv`."""
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
                            with untested_legacy():
                                # Not sure why this never happens anymore
                                primal = engine.resources.convert(primal)
                                res = GraphFunction(primal, Context.empty())
                    else:
                        with untested_legacy():
                            # Not sure why this never happens either
                            res = primal
                            if isinstance(res, Primitive):
                                tid = getattr(f, 'tracking_id', None)
                                res = PrimitiveFunction(res, tracking_id=tid)
                else:
                    raise MyiaTypeError(f'Bad input type for {self.prim}: {f}')
            elif isinstance(f, VirtualFunction):
                res = VirtualFunction(
                    tuple([await self._infer(self, engine, arg)
                           for arg in f.args]),
                    await self._infer(self, engine, f.output.elements[0])
                )
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


@bprop_to_grad_transform(P.Jinv)
def bprop_Jinv(x, out, dout):
    """Backpropagator for primitive `Jinv`."""
    return (J(dout),)


__operation_defaults__ = {
    'name': 'Jinv',
    'registered_name': 'Jinv',
    'mapping': P.Jinv,
    'python_implementation': None,
}


__primitive_defaults__ = {
    'name': 'Jinv',
    'registered_name': 'Jinv',
    'type': 'placeholder',
    'python_implementation': None,
    'inferrer_constructor': infer_Jinv,
    'grad_transform': bprop_Jinv,
}
