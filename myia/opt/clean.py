"""Clean up Class types."""

from ..dtype import Int, Tuple, Class, ismyiatype, type_cloner
from ..ir import Constant, Parameter
from ..prim import ops as P, Primitive
from ..dshape import TupleShape, ClassShape, shape_cloner, NOSHAPE
from ..utils import UNKNOWN


@type_cloner.variant
def _retype(self, t: Class):
    return Tuple[[self(t2) for t2 in t.attributes.values()]]


@shape_cloner.variant
def _reshape(self, s: ClassShape):
    return TupleShape(self(s2) for s2 in s.shape.values())


def erase_class(root, manager):
    """Remove the Class type from graphs.

    * Class[x: t, ...] => Tuple[t, ...]
    * getattr(data, attr) => getitem(data, idx)
    * make_record(cls, *args) => make_tuple(*args)
    """
    manager.add_graph(root)

    for node in list(manager.all_nodes):
        new_node = None

        if node.is_apply(P.getattr):
            _, data, item = node.inputs
            dt = data.type
            assert ismyiatype(dt, Class)
            idx = list(dt.attributes.keys()).index(item.value)
            idx = Constant(idx)
            idx.type = Int[64]
            new_node = node.graph.apply(P.tuple_getitem, data, idx)

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

        if new_node is not None:
            manager.replace(node, new_node)

    for node in manager.all_nodes:
        node.expect_inferred.clear()
        if node.type is not UNKNOWN:
            node.type = _retype(node.type)
        if node.shape is not UNKNOWN:
            node.shape = _reshape(node.shape)


def expand_tuples_p(mng, graph, params):
    new_params = []

    for param in params:
        ttype = param.type
        tshape = param.shape
        if not ismyiatype(ttype, Tuple):
            new_params.append(param)
            continue

        new_param = []
        for te, ts in zip(ttype.elements, tshape.shape):
            np = Parameter(graph)
            np.type = te
            np.shape = ts
            new_param.append(np)

        new_tuple = graph.apply(P.make_tuple, *new_param)
        mng.replace(param, new_tuple)

        new_params.extend(expand_tuples_p(mng, graph, new_param))

    return new_params


def expand_tuples_c(graph, inputs):
    new_inputs = []
    for i in inputs:
        itype = i.type
        ishape = i.shape
        if not ismyiatype(itype, Tuple):
            new_inputs.append(i)
            continue

        new_input = []
        for pos, ie, ish in zip(range(len(itype.elements)),
                                itype.elements, ishape.shape):
            ni = graph.apply(P.tuple_getitem, i, pos)
            ni.inputs[2].type = Int[64]
            ni.inputs[2].shape = NOSHAPE
            new_input.append(ni)

        new_inputs.extend(expand_tuples_c(graph, new_input))

    return new_inputs


def erase_tuple(root, manager):
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
                manager.replace(node, new_node)
        elif (node.is_apply(P.partial) and
              not node.inputs[1].is_constant(Primitive)):
            new_inputs = expand_tuples_c(node.graph,
                                         node.inputs[2:])
            if new_inputs != node.inputs[2:]:
                new_node = node.graph.apply(*node.inputs[:2],
                                            *new_inputs)
                manager.replace(node, new_node)
    # Modify all graph parameters
    for graph in list(manager.graphs):
        manager.set_parameters(graph,
                               expand_tuples_p(manager, graph,
                                               graph.parameters))
