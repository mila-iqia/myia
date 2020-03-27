"""Definitions for the primitive `switch`."""

from .. import operations
from ..lib import (
    Inferrer,
    check_nargs,
    force_pending,
    newenv,
    standard_prim,
    wrap_grad_transform,
)
from ..operations import Jinv, zeros_like
from ..xtype import Bool
from . import primitives as P


def pyimpl_switch(c, x, y):
    """Implement `switch`."""
    return x if c else y


@standard_prim(P.switch)
class _SwitchInferrer(Inferrer):
    """Infer the return type of primitive `switch`."""

    async def reroute(self, engine, outref, argrefs):
        condref, tbref, fbref = check_nargs(P.switch, 3, argrefs)
        cond = await condref.get()
        await force_pending(engine.check(Bool, cond.xtype()))
        v = cond.xvalue()
        if v is True:
            return tbref
        elif v is False:
            return fbref
        else:
            return None

    async def infer(self, engine, cond: Bool, tb, fb):
        return engine.abstract_merge(tb, fb)


@wrap_grad_transform(P.switch)
def __fprop__switch(jcond, jtb, jfb):
    """Backpropagator for primitive `switch`."""
    cond = Jinv(jcond)
    rval = operations.switch(cond, jtb, jfb)

    def __bprop__switch(dout):
        tb = Jinv(jtb)
        fb = Jinv(jfb)
        return (
            newenv,
            zeros_like(cond),
            operations.switch(cond, dout, zeros_like(fb)),
            operations.switch(cond, zeros_like(tb), dout),
        )

    return rval, __bprop__switch


__operation_defaults__ = {
    "name": "switch",
    "registered_name": "switch",
    "mapping": P.switch,
    "python_implementation": pyimpl_switch,
}


__primitive_defaults__ = {
    "name": "switch",
    "registered_name": "switch",
    "type": "backend",
    "python_implementation": pyimpl_switch,
    "inferrer_constructor": _SwitchInferrer,
    "grad_transform": __fprop__switch,
}
