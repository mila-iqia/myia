"""Implements JInferrer."""

from ..dtype import EnvType
from ..prim import ops as P
from ..debug.label import short_relation_symbols as syms

from .graph_infer import Inferrer, ExplicitInferrer, TransformedReference
from .core import reify, reify_shallow


class JInferrer(Inferrer):
    """Inferrer for J(fn).

    Arguments:
        fn: The function to transform.
        mktuple: A function to create a tuple appropriate for the track.
    """

    def __init__(self, fn, mktuple):
        """Initialize a JInferrer."""
        super().__init__(fn.track, 'J')
        self.fn = fn
        assert isinstance(fn, Inferrer)
        self.mktuple = mktuple

    async def infer(self, *jargs):
        """Infer given the arguments."""
        args = [TransformedReference(self.engine, P.Jinv, jarg)
                for jarg in jargs]
        res = await self.fn(*args)
        res_t = self.track.jtag(await reify_shallow(res))
        bparams_t = [EnvType]
        bparams_t += [self.track.stag(await x[self.track.name]) for x in args]
        bprop_t = ExplicitInferrer(
            self.track,
            [self.track.stag(await reify(res))],
            self.mktuple(bparams_t),
            name=f'{syms["grad_bprop"]}{self.fn.identifier}'
        )
        return self.mktuple([res_t, bprop_t])
