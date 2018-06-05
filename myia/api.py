"""User-friendly interfaces to Myia machinery."""

import operator
from types import FunctionType
from typing import Any, Dict, List

from . import parser
from .ir import ANFNode, Graph, clone
from .opt import PatternEquilibriumOptimizer, lib as optlib
from .prim import py_implementations, vm_implementations, ops as P
from .utils import TypeMap
from .vm import VM


def default_object_map() -> Dict[Any, ANFNode]:
    """Get a mapping from Python objects to nodes."""
    mapping = {
        operator.add: P.add,
        operator.sub: P.sub,
        operator.mul: P.mul,
        operator.truediv: P.div,
        operator.mod: P.mod,
        operator.pow: P.pow,
        operator.eq: P.eq,
        operator.ne: P.ne,
        operator.lt: P.lt,
        operator.gt: P.gt,
        operator.le: P.le,
        operator.ge: P.ge,
        operator.pos: P.uadd,
        operator.neg: P.usub,
        operator.not_: P.not_,
        operator.getitem: P.getitem,
        operator.setitem: P.setitem,
        getattr: P.getattr,
        setattr: P.setattr,
    }
    for prim, impl in py_implementations.items():
        mapping[impl] = prim
    return mapping


class Converter:
    """Convert a Python object into an object that can be in a Myia graph."""

    def __init__(self, object_map, converters):
        """Initialize a Converter."""
        self.object_map = object_map
        self.converters = converters

    def __call__(self, value):
        """Convert a value."""
        try:
            return self.object_map[value]
        except (TypeError, KeyError):
            pass

        return self.converters[type(value)](self, value)


def _convert_identity(env, x):
    return x


def _convert_sequence(env, seq):
    return type(seq)(env(x) for x in seq)


def _convert_function(env, fn):
    g = clone(parser.parse(fn))
    env.object_map[fn] = g
    return g


def lax_converter():
    """Return a "lax" converter that allows arbitrary objects."""
    return Converter(
        default_object_map(),
        TypeMap({
            FunctionType: _convert_function,
            tuple: _convert_sequence,
            list: _convert_sequence,
            object: _convert_identity,
            type: _convert_identity,
        })
    )


default_vm = VM(vm_implementations,
                py_implementations,
                lax_converter())


def parse(func: FunctionType, resolve_globals=True) -> Graph:
    """Parse a function into ANF."""
    converter = lax_converter()
    resolver_opt = PatternEquilibriumOptimizer(
        optlib.make_resolver(converter)
    )
    g = converter(func)
    if resolve_globals:
        resolver_opt(g)
        g = clone(g)
    return g


def run(g: Graph, args: List[Any]) -> Any:
    """Evaluate a graph on a set of arguments."""
    return default_vm.evaluate(g, args)


def compile(obj):
    """Return a version of the function that runs using Myia's VM."""
    myia_obj = default_vm.convert(obj)
    return default_vm.export(myia_obj)
