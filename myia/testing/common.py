"""Common testing utilities."""
from ovld import ovld

from myia.abstract import data
from myia.abstract.to_abstract import precise_abstract
from myia.ir.node import Constant, Graph


@precise_abstract.variant
def _to_abstract(self, x: type):
    return data.AbstractAtom({"interface": x})


@ovld
def _to_abstract(self, x: (data.GenericBase, data.AbstractValue)):  # noqa: F811
    return x


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


def Ty(typedesc):pass
def H(typedesc):pass
def af16_of(*shape, value=None):pass
def af32_of(*shape, value=None):pass
def af64_of(*shape, value=None):pass
def ai16_of(*shape, value=None):pass
def ai32_of(*shape, value=None):pass
def ai64_of(*shape, value=None):pass
def au64_of(*shape, value=None):pass
Bool = None
Nil = None
f16 = None
f32 = None
f64 = None
i8 = None
i16 = None
i32 = None
i64 = None
u32 = None
u64 = None


B = Bool

newenv = None
AN = None
JT = None  # AbstractJTagged
TU = None  # AbstractTaggedUnion
Bot = None  # AbstractBottom
D = None  # symbolic dict
EmptyTuple = None
Ex = None
Point = None
Point3D = None
S = None
Shp = None
Thing = None
Thing_f = None
Thing_ftup = None
U = None
mysum = None
to_abstract_test = None
Array = None
EnvType = None
External = None
Int = None
Number = None
String = None
