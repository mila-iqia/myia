from myia.ir import Graph
from myia.operations import Primitive, primitives as P


class NodeDescriptor:
    """Node descriptor class, used to keep inputs and output for a node."""

    PARAM = 0
    CONST = 1
    APPLY = 2
    CLOSURE = 3
    GRAPH = 4

    def __init__(self, type, node):
        """Initialize."""
        self.type = type
        self.node = node
        self.inputs = set()
        self.calls = set()

    def is_valid_j(self):
        """Return True if described node is a J node to expand."""
        return (
            self.type == self.APPLY
            and len(self.inputs) == 2
            and self.node.is_apply(P.J)
            and (
                self.node.inputs[1].is_constant_graph()
                or self.node.inputs[1].is_constant(Primitive)
            )
        )


def describe_node(node, node_to_descriptor=None):
    """Describe a node."""
    if isinstance(node, Graph):
        node_type = NodeDescriptor.GRAPH
        real_node = node.output
    elif node.is_apply():
        node_type = NodeDescriptor.APPLY
        real_node = node
    elif node.is_constant_graph():
        assert not node.inputs
        node_type = NodeDescriptor.GRAPH
        real_node = node.value.output
    elif node.is_constant():
        assert not node.inputs
        node_type = NodeDescriptor.CONST
        real_node = node
    elif node.is_parameter():
        assert not node.inputs
        node_type = NodeDescriptor.PARAM
        real_node = node
    else:
        raise ValueError(f"Unhandled object: {node} of type {type(node)}")

    if node_to_descriptor is not None and real_node in node_to_descriptor:
        return node_to_descriptor[real_node]
    desc = NodeDescriptor(node_type, real_node)
    if node_to_descriptor is not None:
        node_to_descriptor[real_node] = desc
    return desc


def describe_sub_graph(node, calls=None, node_to_descriptor=None, seen=None):
    """Describe sub-graph from a node."""
    calls = calls or ()
    if node_to_descriptor is None:
        node_to_descriptor = {}
    seen = seen or set()
    desc = describe_node(node, node_to_descriptor)
    if desc.type == desc.GRAPH:
        if desc.node.graph in seen:
            return
        seen.add(desc.node.graph)
    for call in calls:
        desc.calls.add(describe_node(call, node_to_descriptor))
    for inp in desc.node.inputs:
        describe_sub_graph(inp, [desc.node], node_to_descriptor, seen)
        desc.inputs.add(describe_node(inp, node_to_descriptor))
    return node_to_descriptor


def lookup_j_in_subgraph(node, debug=False):
    """Check sub-graph from given node and return number of J nodes found."""
    assert node.is_apply(P.J)
    calls = []
    uses = [u[0] for u in node.graph.manager.uses[node]]
    for n in uses:
        if n.is_apply() and n.inputs[0] is node:
            calls.append(n)
    dm = {}
    for call in calls:
        describe_sub_graph(call, node_to_descriptor=dm)
    describe_sub_graph(node, calls, node_to_descriptor=dm)
    inps = [d for d in dm.values() if not d.inputs]
    outs = [d for d in dm.values() if not d.calls]
    js = [d for d in dm.values() if d.is_valid_j() and d.node != node]
    if debug:
        print("Number of first calls", len(calls))
        print("Number of nodes", len(dm))
        print("Number of inputs", len(inps))
        print("Number of outputs", len(outs))
        print("Valid Js", len(js))
        for jn in js:
            print(*jn.node.inputs)
    return len(js)
