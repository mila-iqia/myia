"""Common testing utilities."""
from ovld import ovld

from myia import basics
from myia.abstract import data
from myia.abstract.to_abstract import precise_abstract
from myia.ir.node import Constant, Graph


@precise_abstract.variant
def _to_abstract(self, x: type):
    return data.AbstractAtom({"interface": x})


@ovld
def _to_abstract(self, x: str):  # noqa: F811
    """Keep value for string."""
    return data.AbstractAtom({"interface": str, "value": x})


@ovld
def _to_abstract(self, x: (data.GenericBase, data.AbstractValue)):  # noqa: F811
    return x


@ovld
def _to_abstract(self, x: list):  # noqa: F811
    # Let's expect list to have same type for all list values.
    if not x:
        items = []
    else:
        assert len(x) == 1, x
        (item,) = x
        items = [
            item if isinstance(item, data.AbstractValue) else _to_abstract(item)
        ]
    return data.AbstractStructure(items, {"interface": list})


def A(*args):
    """Convert given arguments to an abstract value for testing."""
    if len(args) == 1:
        arg = args[0]
    else:
        arg = args
    return _to_abstract(arg)


def Un(*opts):
    """Convert given arguments to an abstract union for testing."""
    return data.AbstractUnion([A(opt) for opt in opts], tracks={})


def Ty(element_type):
    """Convert given argument to an abstract type."""
    if element_type is data.ANYTHING:
        return data.AbstractStructure([A(object)], {"interface": type})
    if isinstance(element_type, data.AbstractValue):
        return data.AbstractStructure([element_type], {"interface": type})
    assert isinstance(element_type, type), (element_type, type(element_type))
    return precise_abstract(element_type)


def Aconst(typ, value=data.ANYTHING):
    """Create an abstract constant atom."""
    return data.AbstractAtom({"interface": typ, "value": value})


def H(*opts):
    """Create an abstract handle."""
    return data.AbstractStructure(
        [_to_abstract(opt) for opt in opts], {"interface": basics.Handle}
    )


def D(**kwargs):
    """Create an abstract dictionary."""
    # Warning: does not yet handle types for keys and values.
    return A(kwargs)


def build_node(g, descr, nodeset=set()):
    """Create a node recursively from a tuple of tuples.

    Strings are converted to the parameter with the corresponding name.
    """
    if isinstance(descr, tuple):
        node = g.apply(*[build_node(g, x, nodeset) for x in descr])
    elif isinstance(descr, str):
        (node,) = [p for p in g.parameters if p.name == descr]
    # elif isinstance(descr, dict):
    #     node = g.apply()
    #     for k, v in descr.items():
    #         node.add_edge(k, build_node(g, v, nodeset))
    # elif isinstance(descr, Node):
    #     node = descr
    else:
        node = Constant(descr)
    nodeset.add(node)
    return node


def build_graph(descr, params=[]):
    """Create a graph from a tuple of tuples representing its return value."""
    g = Graph()
    nodeset = set()
    for param_name in params:
        p = g.add_parameter(param_name)
        nodeset.add(p)
    g.output = build_node(g, descr, nodeset)
    nodeset.add(g.return_)
    nodeset.add(g.return_.fn)
    return g, nodeset


def _abstract_sequence(seq_type, *items):
    return data.AbstractStructure(
        [
            item if isinstance(item, data.AbstractValue) else A(item)
            for item in items
        ],
        {"interface": seq_type},
    )


def tuple_of(*items):
    """Create an abstract tuple."""
    return _abstract_sequence(tuple, *items)


def list_of(*items):
    """Create an abstract list."""
    return _abstract_sequence(list, *items)


Object = A(object)
B = Bool = A(bool)
Bot = Nil = A(None)
Float = A(float)
Int = A(int)
Number = Un(int, float)
String = A(str)
EmptyTuple = A(tuple)
