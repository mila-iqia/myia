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
    standard_prim,
)
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
    'grad_transform': None,
}
