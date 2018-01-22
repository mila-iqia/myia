"""User-friendly interfaces to Myia machinery."""
import ast
import operator
from types import FunctionType
from typing import Any, Dict, Type

from myia import parser
from myia.anf_ir import Graph, Constant, ANFNode
from myia.primops import Add, If, Return


ADD = Constant(Add())


def default_ast_map() -> Dict[Type[ast.AST], ANFNode]:
    """Get a mapping from AST binary operations to nodes."""
    return {
        ast.Add: ADD,
        ast.If: Constant(If()),
        ast.Return: Constant(Return())
    }


def default_object_map() -> Dict[Any, ANFNode]:
    """Get a mapping from Python objects to nodes."""
    return {
        operator.add: ADD
    }


ENV = parser.Environment(default_object_map(), default_ast_map())


def parse(func: FunctionType) -> Graph:
    """Parse a function into ANF."""
    return parser.Parser(ENV, func).parse()
