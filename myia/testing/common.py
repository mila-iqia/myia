"""Common testing utilities."""
from dataclasses import dataclass

from ovld import ovld

from myia import basics
from myia.abstract import data
from myia.abstract.to_abstract import precise_abstract
from myia.ir.node import Constant, Graph
from myia.testing import numpy_subset as np


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
    if isinstance(element_type, np.dtype):
        element_type = element_type.type
    assert isinstance(element_type, type), (element_type, type(element_type))
    return precise_abstract(element_type)


def H(*opts):
    """Create an abstract handle."""
    return data.AbstractStructure(
        [_to_abstract(opt) for opt in opts], {"interface": basics.Handle}
    )


def D(**kwargs):
    """Create an abstract dictionary."""
    # Warning: does not yet handle types for keys and values.
    return precise_abstract(kwargs)


def Ex(value, t=None):
    """Abstract external from master branch.

    Just create an abstract value.
    """
    if value is data.ANYTHING:
        return A(object)
    value_type = _to_abstract(value)
    if t is not None:
        assert value_type is t, (value, t, value_type)
    return value_type


def S(x=data.ANYTHING, t=object):
    """Create an abstract scalar."""
    if isinstance(t, data.AbstractAtom):
        t = t.tracks.interface
    assert isinstance(t, type)
    return data.AbstractAtom({"interface": t, "value": x})


def Shp(*dims):
    """Create an abstract tuple representing a shape."""
    return data.AbstractStructure(
        [A(dim) for dim in dims], {"interface": tuple}
    )


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


def _abstract_sequence_getitem(seq_type):
    def getitem(self, *items):
        return _abstract_sequence(seq_type, *items)

    return getitem


def tuple_of(*items):
    """Create an abstract tuple."""
    return _abstract_sequence(tuple, *items)


def list_of(*items):
    """Create an abstract list."""
    return _abstract_sequence(list, *items)


def array_of(dtype: data.AbstractAtom = None, shape=None, value=None):
    """Create an abstract array."""
    items = [dtype] if dtype else []
    tracks = {
        "interface": np.ndarray,
    }
    if shape is not None:
        tracks["shape"] = shape
    if isinstance(shape, tuple):
        tracks["ndim"] = len(shape)
    if value is not None:
        tracks["value"] = value
    return data.AbstractStructure(items, tracks)


def af16_of(*shape, value=None):
    """Create an abstract array of f16."""
    return array_of(f16, shape, value)


def af32_of(*shape, value=None):
    """Create an abstract array of f32."""
    return array_of(f32, shape, value)


def af64_of(*shape, value=None):
    """Create an abstract array of f64."""
    return array_of(f64, shape, value)


def ai16_of(*shape, value=None):
    """Create an abstract array of i16."""
    return array_of(i16, shape, value)


def ai32_of(*shape, value=None):
    """Create an abstract array of i32."""
    return array_of(i32, shape, value)


def ai64_of(*shape, value=None):
    """Create an abstract array of i64."""
    return array_of(i64, shape, value)


def au64_of(*shape, value=None):
    """Create an abstract array of u64."""
    return array_of(u64, shape, value)


Object = A(object)
B = Bool = A(bool)
Bot = Nil = A(None)
f16 = A(np.float16)
f32 = A(np.float32)
f64 = A(np.float64)
i8 = A(np.int8)
i16 = A(np.int16)
i32 = A(np.int32)
i64 = A(np.int64)
u8 = A(np.uint8)
u16 = A(np.uint16)
u32 = A(np.uint32)
u64 = A(np.uint64)

Float = Un(f16, f32, f64)
Integer = Un(i8, i16, i32, i64, u8, u16, u32, u64)
Number = Un(i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64)
Int = {
    8: i8,
    16: i16,
    32: i32,
    64: i64,
}
External = Ex
String = A(str)
EmptyTuple = A(tuple)
EnvType = A(dict)
newenv = {}


def to_abstract_test(x):
    """`to_abstract_test`, from master branch.

    Just make sure given value is an abstract value.
    """
    assert isinstance(x, data.AbstractValue)
    return x


@dataclass(frozen=True)
class Point:
    """Common dataclass for a 2D point."""

    x: i64
    y: i64

    def abs(self):
        """Compute distance from this point to origin."""
        return (self.x ** 2 + self.y ** 2) ** 0.5

    @property
    def absprop(self):
        """Return abs as a property."""
        return self.abs()


@dataclass(frozen=True)
class Point3D:
    """Common dataclass for a 3D point."""

    x: object
    y: object
    z: object

    def abs(self):
        """Compute distance from origin to this point."""
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5


@dataclass(frozen=True)
class Thing:
    """Common dataclass to use for tests."""

    contents: object

    def __call__(self):
        """Overload of call."""
        return self.contents * 2


Thing_f = Thing(1.0)
Thing_ftup = Thing((1.0, 2.0))


@ovld
def mysum(x):  # noqa: F811
    return x


@ovld
def mysum(x, y):  # noqa: F811
    return x + y


@ovld
def mysum(x, y, z):  # noqa: F811
    return x + y + z
