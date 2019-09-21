"""Definitions for the primitive `make_tuple`."""

from ..debug.label import short_relation_symbols as syms
from ..grad import default_grad_flags
from ..lib import AbstractTuple, Graph, MetaGraph, newenv, standard_prim
from . import primitives as P


def pyimpl_make_tuple(*args):
    """Implement `make_tuple`."""
    return args


@standard_prim(P.make_tuple)
async def infer_make_tuple(self, engine, *args):
    """Infer the return type of primitive `make_tuple`."""
    return AbstractTuple(args)


class MakeTupleGradient(MetaGraph):
    """Generate the gradient graph for make_tuple."""

    def generate_graph(self, args):
        """Generate the gradient graph."""
        g = Graph()
        g.debug.name = f'{syms["grad_fprop"]}make_tuple_{len(args)}'

        params = [g.add_parameter() for t in args]
        jinv_params = [g.apply(P.Jinv, p) for p in params]
        tup = g.apply(P.make_tuple, *jinv_params)
        out = g.apply(P.J, tup)

        b = Graph()
        b.debug.name = f'{syms["grad_bprop"]}make_tuple_{len(args)}'
        dout = b.add_parameter()
        grads = [b.apply(P.tuple_getitem, dout, i)
                 for i, p in enumerate(params)]
        b.output = b.apply(P.make_tuple, newenv, *grads)

        g.output = g.apply(P.make_tuple, out, b)
        g.transforms['primal'] = P.make_tuple

        b.flags.update(default_grad_flags)
        g.flags.update(default_grad_flags)

        return g


__operation_defaults__ = {
    'name': 'make_tuple',
    'registered_name': 'make_tuple',
    'mapping': P.make_tuple,
    'python_implementation': pyimpl_make_tuple,
}


__primitive_defaults__ = {
    'name': 'make_tuple',
    'registered_name': 'make_tuple',
    'type': 'backend',
    'python_implementation': pyimpl_make_tuple,
    'inferrer_constructor': infer_make_tuple,
    'grad_transform': MakeTupleGradient(name='make_tuple_gradient'),
}
