
from typing import Any, List, Callable
from types import FunctionType
from numpy import ndarray
from ..util import EventDispatcher, HReprBase
from ..util.debug import Breakpoint
from ..lib import \
    Universe, Closure, Primitive, Function, \
    IdempotentMappable, Record, ZERO, StructuralMap
from ..stx import MyiaASTNode, Symbol, ValueNode, LambdaNode
from ..symbols import builtins, object_map
from ..parse import parse_function
from ..ir.convert import lambda_to_ir


##################################
# Primitive and Function objects #
##################################


class VMPrimitive(Primitive):
    def __init__(self, fn, name, universe):
        super().__init__(fn, name)
        self.universe = universe


class VMFunction(Function):
    def __init__(self, graph, universe):
        ast = graph.lbda
        instrs = make_instructions(graph)
        self.ast = ast
        self.argnames = [a.label for a in ast.args]
        self.args = [n.tag for n in graph.inputs]
        self.graph = graph
        self.universe = universe
        self.code = VMCode(ast, instrs, False)
        self.primal_sym = ast.primal
        self.__myia_graph__ = graph

    def __str__(self):
        return f'VMFunc({self.graph.tag or self.graph})'

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.graph)

    def __eq__(self, other):
        return type(other) is VMFunction \
            and self.graph is other.graph

    def __add__(self, other):
        # TODO: Fix the following issue, which happens sometimes.
        #       I believe the main reason is that for the backpropagator,
        #       Grad builds a closure on the wrong function (?)
        # if self != other:
        #     raise Exception('The functions being added are different.')
        return self

    def __hrepr__(self, H, hrepr):
        return hrepr.titled_box('VMFunc',
                                [hrepr(self.graph.tag or self.graph)])


##########################################
# Code representation for stack-based VM #
##########################################


class Instruction:
    """
    An instruction for the stack-based VM.

    Attributes:
        command: The instruction name.
        node: The Myia node that this instruction is computing.
        args: Instruction-specific arguments.
    """
    def __init__(self,
                 command: str,
                 node: MyiaASTNode,
                 *args: Any) -> None:
        self.command = command
        self.node = node
        self.args = args

    def __str__(self):
        args = ", ".join(map(str, self.args))
        return f'{self.command}:{args}'


def make_instructions(graph):
    instrs = []
    assoc = {}

    def instr(name, node, *args):
        instrs.append(Instruction(name, node, *args))

    def convert(node):
        if node in assoc:
            instr('fetch', node, assoc[node])
        elif node.is_computation():
            succ = node.app()
            for x in succ:
                convert(x)
            instr('reduce', node, len(succ) - 1)
            if len(node.users) > 1:
                # This computation is used more than once, so
                # we store it (and immediately put it back on
                # the stack)
                instr('store', node, node.tag)
                instr('fetch', node, node.tag)
                assoc[node] = node.tag
        elif node.is_builtin():
            assert node.value
            instr('fetch', node, node.value)
        elif node.is_global():
            assert node.value
            instr('fetch', node, node.value)
        elif node.is_graph():
            # raise Exception('Unsupported at the moment')
            instr('fetch', node, node.tag)
        elif node.is_constant():
            instr('push', node, node.value)
        elif node.is_input():
            idx = graph.inputs.index(node)
            assert idx >= 0
            instr('dup', node, idx)
        else:
            raise Exception(f'What is this node? {node}')

    convert(graph.output)
    return instrs


class VMCode(HReprBase):
    """
    Compile a MyiaASTNode into a list of instructions compatible
    with the stack-based VM.

    See VMFrame's ``instruction_<name>`` methods for more
    information.

    Attributes:
        node: The original node.
        instructions: A list of instructions to implement this
            node's behavior.
    """
    def __init__(self,
                 lbda: LambdaNode,
                 instructions: List[Instruction] = None,
                 use_new_ir: bool = True) -> None:
        assert instructions or isinstance(lbda, LambdaNode)
        self.node = None if instructions else lbda.body
        self.lbda = lbda
        self.graph = None
        if instructions is None:
            assert use_new_ir
            self.instructions: List[Instruction] = []
            self.graph = lambda_to_ir(self.lbda).value
            self.instructions = make_instructions(self.graph)
        else:
            self.instructions = instructions

    def __hrepr__(self, H, hrepr):
        rows = []
        for instr in self.instructions:
            row = H.tr()
            for x in (instr.command, *instr.args):
                row = row(H.td(hrepr(x)))
            rows.append(row)
        return H.table['VMCodeInstructions'](*rows)
