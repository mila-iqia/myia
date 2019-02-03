"""Clean up Class types."""

from ..dtype import Int, Tuple, Class, ismyiatype, JTagged
from ..ir import Constant, Parameter
from ..prim import ops as P, Primitive
from ..utils import UNKNOWN
from ..abstract import abstract_clone, AbstractClass, AbstractTuple, \
    AbstractScalar, VALUE, TYPE, SHAPE, ANYTHING, AbstractJTagged


@abstract_clone.variant
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


def expand_tuples_p(mng, graph, params):
    """Expand tuples in graph parameters."""
    new_params = []

    for param in params:
        a = param.abstract
        if not isinstance(a, AbstractTuple):
            if isinstance(a, AbstractJTagged):
                raise NotImplementedError()
            new_params.append(param)
            continue

        new_param = []
        for elem in a.elements:
            np = Parameter(graph)
            np.abstract = elem
            new_param.append(np)

        new_tuple = graph.apply(P.make_tuple, *new_param)
        mng.replace(param, new_tuple)

        new_params.extend(expand_tuples_p(mng, graph, new_param))

    return new_params


def expand_tuples_c(graph, inputs):
    """Expand tuples in graph applies."""
    new_inputs = []
    for i in inputs:
        a = i.abstract
        if not isinstance(a, AbstractTuple):
            if (isinstance(a, AbstractJTagged) and
                    isinstance(a.element, AbstractTuple)):
                raise NotImplementedError("JTagged")
            else:
                new_inputs.append(i)
                continue

        new_input = []

        for pos, elem in enumerate(a.elements):
            ni = graph.apply(P.tuple_getitem, i, pos)
            ni.inputs[2].abstract = AbstractScalar({
                VALUE: pos,
                TYPE: Int[64],
            })
            ni.abstract = elem
            new_input.append(ni)

        new_inputs.extend(expand_tuples_c(graph, new_input))

    return new_inputs


def erase_tuple(root, manager):
    """Remove most uses of tuples from the graph.

    tuples that are returned will be kept.
    """
    manager.add_graph(root)

    # Fix up all call sites (except primitives)
    for node in list(manager.all_nodes):
        if (node.is_apply() and
                not node.inputs[0].is_constant(Primitive)):
            new_inputs = expand_tuples_c(node.graph,
                                         node.inputs[1:])
            if new_inputs != node.inputs[1:]:
                new_node = node.graph.apply(node.inputs[0],
                                            *new_inputs)
                new_node.abstract = node.abstract
                manager.replace(node, new_node)
        elif (node.is_apply(P.partial) and
              not node.inputs[1].is_constant(Primitive)):
            new_inputs = expand_tuples_c(node.graph,
                                         node.inputs[2:])
            if new_inputs != node.inputs[2:]:
                new_node = node.graph.apply(*node.inputs[:2],
                                            *new_inputs)
                new_node.abstract = node.abstract
                manager.replace(node, new_node)
    # Modify all graph parameters
    for graph in list(manager.graphs):
        manager.set_parameters(graph,
                               expand_tuples_p(manager, graph,
                                               graph.parameters))
