"""Implementation of the 'array_reduce_dim' operation."""

from ..lib import SHAPE, Graph, MetaGraph, MyiaTypeError, build_value
from ..operations import primitives as P


class ArrayReduceDim(MetaGraph):
    """MetaGraph to reduce array over specified dim(s)."""

    def __init__(self, name="ArrayReduceDim"):
        """Initialize a ArrayReduceDim."""
        super().__init__(name)

    def generate_graph(self, args):
        """Generate the graph."""
        g = Graph()
        for arg in args:
            g.add_parameter()
        x = g.parameters[1]
        orig_shp = g.apply(P.shape, x)
        try:
            dim = build_value(args[2])
        except ValueError:
            raise MyiaTypeError("Dimension reduction must be known at "
                                "compile time.")
        if not isinstance(dim, tuple):
            dim = (dim,)
        new_shp_unsqueezed = orig_shp
        for d in dim:
            new_shp_unsqueezed = g.apply(P.tuple_setitem, new_shp_unsqueezed,
                                         d, 1)
        array_squash = g.apply(
            P.array_reduce, g.parameters[0], x, new_shp_unsqueezed)
        try:
            keepdim = build_value(args[3])
        except ValueError:
            raise MyiaTypeError("Keepdim must be known at "
                                "compile time.")
        g_output = array_squash
        if not keepdim:
            f_s = []
            for i in range(len(args[1].values[SHAPE])):
                if i not in dim:
                    f_s.append(g.apply(P.tuple_getitem, orig_shp, i))
            final_shape = g.apply(P.make_tuple, *f_s)
            array_reduced = g.apply(P.reshape, array_squash, final_shape)
            g_output = array_reduced
        g.output = g_output
        return g


array_reduce_dim = ArrayReduceDim()


__operation_defaults__ = {
    'name': 'array_reduce_dim',
    'registered_name': 'array_reduce_dim',
    'mapping': array_reduce_dim,
    'python_implementation': None,
}
