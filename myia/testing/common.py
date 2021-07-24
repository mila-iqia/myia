"""Common testing utilities."""
from dataclasses import dataclass
import numpy as np
from ovld import ovld
from myia import basics

from myia.abstract import data
from myia.abstract.to_abstract import precise_abstract
from myia.ir.node import Constant, Graph


class _TupleFactory(data.AbstractStructure):
    __type__ = tuple

    def __init__(self, *items):
        super().__init__([item if isinstance(item, data.AbstractValue) else A(item) for item in items], {"interface": self.__type__})

    def __getitem__(self, *items):
        return type(self)(*items)


class _ListFactory(_TupleFactory):
    __type__ = list


class _ArrayFactory(_TupleFactory):
    __type__ = np.ndarray

    def __iter__(self, *items: data.AbstractAtom):
        assert len(items) < 2
        super().__init__(*items)

    def __getitem__(self, item: data.AbstractAtom):
        return super().__getitem__(item)

    def of(self, scalar_type: data.AbstractAtom, shape, value=None):
        tracks = {
            "interface": np.ndarray,
            "ndim":len(shape),
            "shape": tuple(shape),
        }
        if value is not None:
            tracks["value"] = value
        return data.AbstractStructure([scalar_type], tracks)


class _ExternalFactory(data.AbstractAtom):
    def __init__(self):
        super().__init__({"interface": object})

    def __getitem__(self, item):
        return A(item)


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


def Ty(element_type):
    if element_type is data.ANYTHING:
        return data.AbstractStructure([A(object)], {"interface": type})
    if isinstance(element_type, data.AbstractValue):
        return data.AbstractStructure([element_type], {"interface": type})
    if isinstance(element_type, np.dtype):
        element_type = element_type.type
    assert isinstance(element_type, type), (element_type, type(element_type))
    return precise_abstract(element_type)


def H(*opts):
    return data.AbstractStructure(
        [precise_abstract(opt) for opt in opts],
        {"interface": basics.Handle}
    )


def D(**kwargs):
    """Abstract dictionary."""
    # Warning: does not yet handle types for keys and values.
    return precise_abstract(kwargs)


def Ex(value, t=None):
    """Abstract external from master branch"""
    if value is data.ANYTHING:
        return A(object)
    value_type = _to_abstract(value)
    if t is not None:
        assert value_type is t, (value, t, value_type)
    return value_type


def S(x=data.ANYTHING, t=object):
    if isinstance(t, data.AbstractAtom):
        t = t.tracks.interface
    assert isinstance(t, type)
    return data.AbstractAtom({"interface": t, "value": x})


def Shp(*dims):
    return data.AbstractStructure([A(dim) for dim in dims], {"interface": tuple})


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


def af16_of(*shape, value=None): return Array.of(f16, shape, value)
def af32_of(*shape, value=None):return Array.of(f32, shape, value)
def af64_of(*shape, value=None):return Array.of(f64, shape, value)
def ai16_of(*shape, value=None):return Array.of(i16, shape, value)
def ai32_of(*shape, value=None):return Array.of(i32, shape, value)
def ai64_of(*shape, value=None):return Array.of(i64, shape, value)
def au64_of(*shape, value=None):return Array.of(u64, shape, value)


B = Bool = A(bool)
Bot = Nil = A(None)
f16 = A(np.float16)
f32 = A(np.float32)
f64 = A(np.float64)
i8 = A(np.int8)
i16 = A(np.int16)
i32 = A(np.int32)
i64 = A(np.int64)
u32 = A(np.uint32)
u64 = A(np.uint64)

Int = {
    8: i8,
    16: i16,
    32: i32,
    64: i64,
}
Tuple = _TupleFactory()
List = _ListFactory()
Array = _ArrayFactory()
External = _ExternalFactory()

U = Un


def to_abstract_test(x):
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


class _Nothing:
    """Temporar placeholder."""
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"{type(self).__name__}({self.name})"

    __repr__ = __str__

    def __getitem__(self, item):
        raise TypeError(f"Cannot get {self}[{item}]")

    def __call__(self, *args, **kwargs):
        raise TypeError(f"Cannot call {self}")


newenv = _Nothing("newenv")
AN = _Nothing("AN")
EmptyTuple = _Nothing("EmptyTuple")
Thing_f = _Nothing("Thing_f")
Thing_ftup = _Nothing("Thing_ftop")
mysum = _Nothing("mysum")
EnvType = _Nothing("EnvType")
Number = _Nothing("Number")
String = _Nothing("String")
