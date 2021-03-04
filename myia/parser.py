import ast
import inspect

import textwrap
from typing import NamedTuple

from .ir import Node, Apply, Constant, Graph, Parameter
from .ir.node import SEQ
from .utils import ClosureNamespace, ModuleNamespace


class Location(NamedTuple):
    """A location in source code.
    Attributes:
        filename: The filename.
        line: The line number.
        column: The column number.
    """

    filename: str
    line: int
    column: int
    line_end: int
    column_end: int


class MyiaSyntaxError(Exception):
    """Exception to indicate that the syntax is invalid for myia."""

    def __init__(self, msg, loc):
        """Initialize with a message and source location."""
        super().__init__(msg)
        self.loc = loc


ast_map = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "truediv",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Pow: "pow",
    ast.MatMult: "matmul",
    ast.LShift: "lshift",
    ast.RShift: "rshift",
    ast.BitAnd: "and_",
    ast.BitOr: "or_",
    ast.BitXor: "xor",
    ast.UAdd: "pos",
    ast.USub: "neg",
    ast.Invert: "invert",
    ast.Not: "not_",
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Lt: "lt",
    ast.Gt: "gt",
    ast.LtE: "le",
    ast.GtE: "ge",
    ast.Is: "is_",
    ast.IsNot: "is_not",
    ast.In: "contains",
    ast.NotIn: "not_in",
}


_parse_cache = {}


def parse(func):
    if func in _parse_cache:
        return _parse_cache[key]

    flags = dict(getattr(func, "_myia_flags", {}))

    if "name" in flags:
        name = flags["name"]
        del flags["name"]
    else:
        name = None

    inner_flags = {
        flag: True for flag, value in flags.items() if value == "inner"
    }
    flags.update(inner_flags)

    parser = Parser(func, recflags=flags)
    graph = parser.parse()

    for flag in inner_flags:
        del graph.flags[flag]

    if name is not None:
        graph.name = name
    _parse_cache[func] = graph
    return graph


class Block:
    def __init__(self, parent, name, location=None, flags={}):
        self.parent = parent
        self.preds = set()

        if self.parent:
            pgraph = self.parent.graph
        else:
            pgraph = None
        self.graph = Graph(pgraph)
        self.graph.name = name
        self.graph.location = location
        self.graph.set_flags(reference=True)
        self.graph.flags.update(flags)

        self.last_apply = None
        self.var_seq = 0
        self.variables_read = {}
        self.variables_written = {}

    def write(self, varnum, node):
        st = self.graph.apply('store', varnum, node, self.var_seq)
        self.var_seq += 1
        self.variables_written.setdefault(varnum, []).append(st)
        return st

    def read(self, varnum):
        ld = self.graph.apply("load", varnum, self.var_seq)
        self.var_seq += 1
        self.variables_read.setdefault(varnum, []).append(ld)
        return ld

    def apply(self, *inputs):
        return self.graph.apply(*inputs)

    def link_seq(self, node):
        node.add_edge(SEQ, self.last_apply)
        self.last_apply = node
        return node

    def returns(self, value):
        assert self.graph.return_ is None
        self.graph.output = value
        self.link_seq(self.graph.return_)


class Parser:
    def __init__(self, function, recflags):
        self.function = function
        self.recflags = recflags
        src0 = inspect.getsource(self.function)
        self.src = textwrap.dedent(src0)
        self.col_offset = src0.index("\n") - self.src.index("\n")
        _, self.line_offset = inspect.getsourcelines(function)
        self.line_offset -= 1
        self.filename = inspect.getfile(function)
        self.global_namespace = ModuleNamespace(function.__module__)
        self.closure_namespace = ClosureNamespace(function)
        self.finalizers = {}

    def _eval_ast_node(self, node):
        text = ast.get_source_segment(self.src, node)
        # XXX: This needs the locals at the point of the annotation
        # to match "old" python behaviour.  The closure_namespace doesn't
        # have what we need since python doesn't keep a closure for
        # annotations.

        # We currently act as if "from __future__ import annotations" is
        # in effect, which means that we only need to care about globals.
        return eval(text, self.function.__globals__)

    def new_block(self, parent, name, location, flags={}):
        return Block(parent, name, location, dict(**self.recflags, **flags))

    def make_location(self, node):
        if node is None:
            return None
        if isinstance(node, (list, tuple)):
            if len(node) == 0:
                return None
            node0 = node[0]
            node1 = node[-1]
        else:
            node0 = node
            node1 = node
        if hasattr(node0, "lineno") and hasattr(node1, "end_col_offset"):
            li1 = node0.lineno + self.line_offset
            col1 = node0.col_offset + self.col_offset
            li2 = node1.end_lineno + self.line_offset
            col2 = node1.end_col_offset + self.col_offset
            return Location(self.filename, li1, col1, li2, col2)
        else:
            # Some nodes like Index carry no location information, but
            # we basically just pass through them.
            return None

    def parse(self):
        tree = ast.parse(self.src, filename=self.filename)
        function_def = tree.body[0]
        assert isinstance(function_def, ast.FunctionDef)

        main_block = self._create_function(None, function_def)

        # Check for no return
        # This does a dfs and finds graphs with None for return_

        return main_block.graph

    def _create_function(self, block, node):
        function_block = self.new_block(parent=block,
                                        name=node.name,
                                        location=self.make_location(node))
        args = node.args.args
        nondefaults = [None] * (len(args) - len(node.args.kw_defaults))
        defaults = nondefaults + node.args.defaults

        kwargs = node.args.kwonlyargs
        kwnondefaults = [None] * (len(kwargs) - len(node.args.kw_defaults))
        kwdefaults = kwnondefaults + node.args.kw_defaults

        defaults_name = []
        defaults_list = []

        for arg, dflt in zip(args + kwargs, defaults + kwdefaults):
            param_node = Parameter(function_block.graph, name=arg.arg,
                                   location=self.make_location(arg))

            if arg.annotation:
                param_node.add_annotation(self._eval_ast_node(arg.annotation))

            function_block.graph.parameters.append(param_node)
            function_block.write(arg.arg, param_node)
            if dflt:
                dflt_node = self.process_node(function_block, dflt)
                defaults_names.append(arg.arg)
                defaults_list.append(dflt_node)

        if node.args.vararg:
            arg = node.args.vararg
            vararg_node = Parameter(function_block.graph, name=arg.arg,
                                    location=self.make_location(arg))
            function_block.graph.parameters.append(vararg_node)
            function_block.write(arg.arg, vararg_node)
        else:
            vararg_node = None

        if node.args.kwarg:
            arg = node.args.kwarg
            kwarg_node = Parameter(function_block.graph, name=arg.arg,
                                   location=self.make_location(arg))
            function_block.graph.parameters.append(kwarg_node)
            function_block.write(arg.arg, kwarg_node)
        else:
            kwarg_node = None

        self.process_statements(function_block, node.body)

        if function_block.graph.return_ is None:
            raise MyiaSyntaxError(
                "Function doesn't return a value", self.make_location(node)
            )

        function_block.graph.vararg = vararg_node and vararg_node.name
        function_block.graph.kwarg = kwarg_node and kwarg_node.name
        function_block.graph.defaults = dict(zip(defaults_name,
                                                 defaults_list))
        function_block.graph.kwonly = len(node.args.kwonlyargs)

        if node.returns:
            function_block.graph.return_.add_annotation(self._eval_ast_node(node.returns))

        return function_block

    def make_condition_blocks(self, block, tn, fn):
        tb = self.new_block(block, "if_true", self.make_location(tn))
        tb.preds.add(block)

        fb = self.new_block(block, "if_false", self.make_location(fn))
        fb.preds.add(block)

        return tb, fb

    # expressions (returns a value)

    def process_node(self, block, node):
        method_name = f"process_{node.__class__.__name__}"
        method = getattr(self, method_name, None)
        if method:
            return method(block, node)
        else:
            raise MyiaSyntaxError(
                f"{node.__class__.__name__} not supported",
                self.make_location(node)
            )

    def process_Name(self, block, node):
        assert isinstance(node.ctx, ast.Load)
        return block.read(node.id)

    def process_BinOp(self, block, node):
        return block.apply(ast_map[type(node.op)],
                           self.process_node(block, node.left),
                           self.process_node(block, node.right))

    def _fold_bool(self, block, values, mode):
        first, *rest = values
        test = self.process_node(block, first)
        if rest:
            if mode == "and":
                tb, fb = self.make_condition_blocks(block, rest, None)
                fb.returns(Constant(False))
                tb.returns(self._fold_bool(tb, rest, mode))
            else:
                tb, fb = self.make_condition_blocks(block, None, rest)
                tb.returns(Constant(True))
                fb.returns(self._fold_bool(fb, rest, mode))
            switch = block.apply("switch", test, tb.graph, fb.graph)
            return block.apply(switch)
        else:
             return test

    def process_BoolOp(self, block, node):
        if isinstance(node.op, ast.And):
            return self._fold_bool(block, node.values, "and")
        elif isinstance(node.op, ast.Or):
            return self._fold_bool(block, node.values, "or")
        else:
            raise AssertionError(f"Unknown BoolOp: {node.op}")

    def process_Compare(self, block, node):
        if len(node.ops) == 1:
            left = self.process_node(block, node.left)
            right = self.process_node(block, node.comparators[0])
            return block.apply(ast_map[type(node.ops[0])], left, right)
        else:
            cur = node.left
            rest = node.comparators
            ops = node.ops
            values = []
            while ops:
                # TODO: fix up source locations
                values.append(ast.Compare(ops=[ops[0]], left=cur, comparators=[rest[0]]))
                cur, rest = rest[0], rest[1:]
                ops = ops[1:]
            return self._fold_bool(block, values, "and")

    def process_Constant(self, block, node):
        return Constant(node.value)

    def process_IfExp(self, block, node):
        cond = self.process_node(block, node.test)
        tb, fb = self.make_condition_blocks(block, node.body, node.orelse)

        tn = self.process_node(tb, node.body)
        fn = self.process_node(fb, node.orelse)

        tb.returns(tn)
        fb.returns(fn)

        switch = block.apply("user_switch", cond, tb.graph, fb.graph)
        return block.apply(switch)

    def process_UnaryOp(self, block, node):
        val = self.process_node(block, node.operand)
        return block.apply(ast_map[type(node.op)], val)

    # statements (returns a block)

    def process_statements(self, starting_block, nodes):
        block = starting_block
        for node in nodes:
            block = self.process_node(block, node)
        return block

    def _assign(self, block, targ, val):
        if isinstance(targ, ast.Name):
            # x = val
            st = block.write(targ.id, val)
            block.link_seq(st)

        elif isinstance(targ, ast.Tuple):
            # x, y = val
            for i, elt in enumerate(targ.elts):
                nn = block.apply("getitem", node, i)
                self._assign(block, elt, nn)

        else:
            raise NotImplementedError(targ)

    def process_Assign(self, block, node):
        val = self.process_node(block, node.value)
        for targ in node.targets:
            self._assign(block, targ, val)

        return block

    def process_Pass(self, block, node):
        return block

    def process_Return(self, block, node):
        block.returns(self.process_node(block, node.value))
        return block
