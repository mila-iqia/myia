"""User-friendly interfaces to Myia machinery."""
import operator
from types import FunctionType
from typing import Any, Callable, Dict, List, Union

from . import parser
from .ir import ANFNode, Graph
from .opt import pattern_equilibrium_optimizer, lib as optlib
from .prim import py_implementations, vm_implementations, ops as P, Primitive
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


restricted_types = (int, float, bool, str, Primitive)
all_types = (object,)
default_vm = VM(vm_implementations,
                py_implementations,
                default_object_map(),
                all_types)
resolver_opt = pattern_equilibrium_optimizer(optlib.make_resolver(default_vm))


def parse(func: FunctionType, resolve_globals=True) -> Graph:
    """Parse a function into ANF."""
    g = parser.parse(func)
    if resolve_globals:
        resolver_opt(g)
    return g


def run(g: Graph, args: List[Any]) -> Any:
    """Evaluate a graph on a set of arguments."""
    return default_vm.evaluate(g, args)


def compile(func: Union[Graph, FunctionType]) -> Callable:
    """Return a version of the function that runs using Myia's VM."""
    if not isinstance(func, Graph):
        func = parse(func)
    return default_vm.make_callable(func)
