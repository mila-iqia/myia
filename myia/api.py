"""User-friendly interfaces to Myia machinery."""
import operator
from types import FunctionType
from typing import Any, Callable, Dict, List, Union

from . import parser
from .ir import ANFNode, Constant, Graph
from .prim import implementations, ops as P
from .vm import VM as VM_


def default_object_map() -> Dict[Any, ANFNode]:
    """Get a mapping from Python objects to nodes."""
    mapping: Dict[Any, ANFNode] = {
        operator.add: Constant(P.add),
        operator.sub: Constant(P.sub),
        operator.mul: Constant(P.mul),
        operator.truediv: Constant(P.div),
        operator.mod: Constant(P.mod),
        operator.pow: Constant(P.pow),
        operator.eq: Constant(P.eq),
        operator.ne: Constant(P.ne),
        operator.lt: Constant(P.lt),
        operator.gt: Constant(P.gt),
        operator.le: Constant(P.le),
        operator.ge: Constant(P.ge),
        operator.pos: Constant(P.uadd),
        operator.neg: Constant(P.usub),
        operator.not_: Constant(P.not_),
        operator.getitem: Constant(P.getitem),
        operator.setitem: Constant(P.setitem),
        getattr: Constant(P.getattr),
        setattr: Constant(P.setattr)
    }
    for prim, impl in implementations.items():
        mapping[impl] = Constant(prim)

    return mapping


ENV = parser.Environment(default_object_map().items())
VM = VM_(implementations)


def parse(func: FunctionType) -> Graph:
    """Parse a function into ANF."""
    return ENV.map(func).value


def run(g: Graph, args: List[Any]) -> Any:
    """Evaluate a graph on a set of arguments."""
    return VM.evaluate(g, args)


def compile(func: Union[Graph, FunctionType]) -> Callable:
    """Return a version of the function that runs using Myia's VM."""
    if not isinstance(func, Graph):
        func = parse(func)
    return VM.make_callable(func)
