
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
from ..ir.graph import IRGraph
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
        self.ast = ast
        self.argnames = [a.label for a in ast.args]
        self.args = [n.tag for n in graph.inputs]
        self.graph = graph
        self.universe = universe
        self.code = VMCode(graph)
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
    stack_size = len(graph.inputs)

    order = [node for node in graph.toposort()
             if len(node.users) > 1]

    def instr(name, node, *args):
        instrs.append(Instruction(name, node, *args))

    def convert(node, top=False):
        nonlocal stack_size
        if node in assoc:
            instr('dup', node, assoc[node])
            stack_size += 1
        elif node.is_computation():
            succ = node.app()
            assert all(node for node in succ)
            for x in succ:
                convert(x)
            nargs = len(succ) - 1
            instr('reduce', node, nargs)
            stack_size -= nargs
            if len(node.users) > 1:
                # Sanity check. Bad things will happen if this fails.
                assert top
        elif node.is_builtin():
            assert node.value
            instr('fetch', node, node.value)
            stack_size += 1
        elif node.is_global():
            assert node.value
            instr('fetch', node, node.value)
            stack_size += 1
        elif node.is_graph():
            instr('fetch', node, node.tag)
            stack_size += 1
        elif node.is_constant():
            instr('push', node, node.value)
            stack_size += 1
        elif node.is_input():
            idx = graph.inputs.index(node)
            assert idx >= 0
            instr('dup', node, idx)
            stack_size += 1
        else:
            raise Exception(f'What is this node? {node}')

    for node in order:
        convert(node, True)
        assoc[node] = stack_size - 1

    convert(graph.output, True)

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
                 graph: IRGraph,
                 instructions: List[Instruction] = None) -> None:
        self.graph = graph
        self.lbda = graph.lbda
        self.node = None if instructions else self.lbda.body
        if instructions is None:
            self.instructions: List[Instruction] = []
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
