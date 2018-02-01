"""User-friendly interfaces to Myia machinery."""
import ast
import operator
from types import FunctionType
from typing import Any, Dict, List, Type

from myia import parser
from myia.anf_ir import Graph, Constant, ANFNode
from myia import primops as P
from myia.vm import VM as VM_
from myia.py_implementations import implementations


def default_ast_map() -> Dict[Type[ast.AST], ANFNode]:
    """Get a mapping from AST binary operations to nodes."""
    return {
        ast.Add: Constant(P.add),
        ast.Sub: Constant(P.sub),
        ast.Mult: Constant(P.mul),
        ast.Div: Constant(P.div),
        ast.Mod: Constant(P.mod),
        ast.Pow: Constant(P.pow),
        ast.UAdd: Constant(P.uadd),
        ast.USub: Constant(P.usub),
        ast.Eq: Constant(P.eq),
        ast.NotEq: Constant(P.ne),
        ast.Lt: Constant(P.lt),
        ast.Gt: Constant(P.gt),
        ast.LtE: Constant(P.le),
        ast.GtE: Constant(P.ge),
        ast.Not: Constant(P.not_),
        ast.Tuple: Constant(P.make_tuple),
        ast.Subscript: Constant(P.getitem),
        ast.Attribute: Constant(P.getattr),
        ast.If: Constant(P.if_),
        ast.Return: Constant(P.return_)
    }


def default_object_map() -> Dict[int, ANFNode]:
    """Get a mapping from Python objects to nodes."""
    mapping = {
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
        operator.not_: Constant(P.not_),
        operator.getitem: Constant(P.getitem),
        operator.setitem: Constant(P.setitem),
        getattr: Constant(P.getattr),
        setattr: Constant(P.setattr)
    }
    for prim, impl in implementations.items():
        mapping[impl] = Constant(prim)

    return {id(k): v for k, v in mapping.items()}


ENV = parser.Environment(default_object_map(), default_ast_map())
VM = VM_(implementations)


def parse(func: FunctionType) -> Graph:
    """Parse a function into ANF."""
    return parser.Parser(ENV, func).parse()


def run(g: Graph, args: List[Any]) -> Any:
    """Evaluate a graph on a set of arguments."""
    return VM.evaluate(g, args)
