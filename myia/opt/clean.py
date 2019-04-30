"""Clean up Class types."""

from ..dtype import Int
from ..ir import Constant, Parameter, Graph
from ..prim import ops as P
from ..abstract import abstract_clone, AbstractClass, AbstractTuple, \
    AbstractScalar, VALUE, TYPE, AbstractJTagged


@abstract_clone.variant(wrapper=None)
def _reabs(self, a: AbstractClass):
    return AbstractTuple(self(x) for x in a.attributes.values())


def erase_class(root, manager):
    """Remove the Class type from graphs.

    * Class[x: t, ...] => Tuple[t, ...]
    * getattr(data, attr) => getitem(data, idx)
    * make_record(cls, *args) => make_tuple(*args)
    """
    from ..abstract import AbstractClass
    from ..utils import is_dataclass_type

    manager.add_graph(root)

    for node in list(manager.all_nodes):
        new_node = None

        if node.is_apply(P.getattr):
            _, data, item = node.inputs
            dt = data.abstract
            assert isinstance(dt, AbstractClass)
            idx = list(dt.attributes.keys()).index(item.value)
            idx_c = Constant(idx)
            idx_c.abstract = AbstractScalar({
                VALUE: idx,
                TYPE: Int[64],
            })
            new_node = node.graph.apply(P.tuple_getitem, data, idx_c)

        elif node.is_apply(P.make_record):
            mkr, typ, *args = node.inputs
            new_node = node.graph.apply(P.make_tuple, *args)

        elif node.is_apply(P.partial):
            orig_ptl, oper, *args = node.inputs
            if oper.is_constant() and oper.value is P.make_record:
                if len(args) == 1:
                    new_node = Constant(P.make_tuple)
                elif len(args) > 1:
                    new_node = node.graph.apply(
                        P.partial, P.make_tuple, *args[1:]
                    )

        elif node.is_constant() and is_dataclass_type(node.value):
            new_node = Constant(P.make_tuple)

        if new_node is not None:
            new_node.abstract = node.abstract
            manager.replace(node, new_node)

    for node in manager.all_nodes:
        node.abstract = _reabs(node.abstract)


def params_no_tuple(params):
    """Make a new version of params without tuples.

    This will return the new parameters and a list of node to rebuild
    the original arguments from the new parameters.
    """
    g = Graph()
    inputs = []

    def _helper(p):
        a = p.abstract
        if not isinstance(a, AbstractTuple):
            assert not isinstance(a, AbstractJTagged)
            np = g.add_parameter()
            np.abstract = a
            return np

        new_param = []
        for elem in a.elements:
            np = Parameter(g)
            np.abstract = elem
            new_param.append(_helper(np))
        return g.apply(P.make_tuple, *new_param)

    inputs = [_helper(p) for p in params]
    return g, inputs


def erase_tuple(root, manager):
    """Remove use of tuple in the main function arguments."""
    params = root.parameters

    if any(isinstance(p.abstract, AbstractTuple) for p in params):
        g, inputs = params_no_tuple(params)
        g.output = g.apply(root, *inputs)
        # Set the new graph as the entry point
        manager.keep_roots(g)
        return g

    return root
