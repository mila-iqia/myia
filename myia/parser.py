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
        return _parse_cache[func]

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
    def __init__(self, function, name, parent_graph, location=None, flags={}):
        self.function = function

        self.graph = Graph(parent_graph)
        self.graph.name = name
        self.graph.location = location
        self.graph.set_flags(reference=True)
        self.graph.flags.update(flags)

        self.jump_tag = None
        self.last_apply = None
        self.used = True

        self.variables_written = {}
        self.variables_read = {}

    def apply(self, *inputs):
        res = self.graph.apply(*inputs)
        self.link_seq(res)
        return res

    def read(self, varnum):
        st = self.variables_written.get(varnum, [None])[-1]
        ld = self.apply("load", varnum, st)
        self.variables_read.setdefault(varnum, []).append(ld)
        if not varnum in self.function.variables_local:
            self.function.variables_free.add(varnum)
        return ld

    def write(self, varnum, node):
        st = self.apply('store', varnum, node)
        self.variables_written.setdefault(varnum, []).append(st)
        self.function.variables_local.add(varnum)
        self.function.variables_first_write.setdefault(varnum, st)
        return st

    def link_seq(self, node):
        node.add_edge(SEQ, self.last_apply)
        self.last_apply = node
        return node

    def cond(self, cond, true, false):
        assert self.graph.return_ is None
        switch = self.apply("user_switch", cond, true.graph, false.graph)
        self.returns(self.apply(switch))

    def jump(self, target, *args):
        assert self.graph.return_ is None
        jump = self.apply(target.graph, *args)
        self.jump_tag = (target, jump)
        self.returns(jump)

    def returns(self, value):
        assert self.graph.return_ is None
        self.graph.output = value
        self.link_seq(self.graph.return_)


class Function:
    def __init__(self, parent, name, location=None, flags={}):
        if parent is not None:
            parent_function = parent.function
            parent_graph = parent.graph
        else:
            parent_function = None
            parent_graph = None
        self.parent = parent_function
        self.flags = flags
        self.children = set()
        if self.parent is not None:
            self.parent.children.add(self)

        self.initial_block = Block(self, name, parent_graph,
                                   location, flags)
        self.blocks = [self.initial_block]

        self.variables_free = set()
        self.variables_root = set()
        self.variables_local_closure = set()

        self.variables_global = set()
        self.variables_local = set()

        self.variables_nonlocal = set()
        self.variables_first_write = dict()

    def new_block(self, name, parent, location=None):
        block = Block(self, name, parent.graph, location,
                      self.flags)
        self.blocks.append(block)
        return block


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

        # Order from parents to children
        functions = self.all_functions(main_block.function)
        self.analyze(functions)
        self.resolve(functions)

        # Check for no return
        # This does a dfs and finds graphs with None for return_

        return main_block.graph

    def all_functions(self, main_function):
        functions = [main_function]
        todo = list(main_function.children)
        while todo:
            fn = todo.pop()
            functions.append(fn)
            todo.extend(fn.children)
        return functions

    def analyze(self, functions):
        for function in reversed(functions):
            # remove from locals what was marked as not a local
            function.variables_local -= function.variables_nonlocal
            function.variables_local -= function.variables_global

            function.variables_free -= function.variables_global
            function.variables_free.update(function.variables_nonlocal)

            for err in function.variables_free.intersection(
                function.variables_local
            ):
                # If this intersection is non-empty it means
                # that we reference the variables it contains
                # before we assign to them
                raise UnboundLocalError(f"local variable '{err}' is referenced before assignment")

            for child in function.children:
                # variables_free is the sum of all free variables
                # including children
                function.variables_free.update(
                    child.variables_free - function.variables_local
                )
                # variables_root is the set of closure variables defined here
                function.variables_root.update(
                    function.variables_local.intersection(child.variables_free)
                )

            for block in function.blocks:
                if block is function.initial_block or block.used is False:
                    continue
                for var, reads in block.variables_read.items():
                    if reads[0].edges[1].node.is_constant():
                        function.variables_local_closure.add(var)

    def resolve_read(self, repl, repl_seq, ld, function, local_namespace):
        var = ld.edges[0].node.value
        st = ld.edges[1].node
        if var in (function.variables_root | function.variables_free |
                   function.variables_local_closure):
            if var not in local_namespace:
                n = ld.graph.apply("resolve", self.global_namespace, var)
            else:
                n = ld.graph.apply("universe_getitem", local_namespace[var])
            n.edges[SEQ] = ld.edges[SEQ]
            repl[ld] = n
        elif var in function.variables_global:
            n = ld.graph.apply("resolve", self.global_namespace, var)
            n.edges[SEQ] = ld.edges[SEQ]
            repl[ld] = n
        elif var in function.variables_local:
            if st.is_apply():
                repl[ld] = st.edges[1].node
            else:
                # st can be none if it wasn't written to in the block namespace
                repl[ld] = local_namespace[var]
            repl_seq[ld] = ld.edges[SEQ].node
        else:
            raise AssertionError(f"unclassified variable '{var}'")

    def resolve_write(self, repl, repl_seq, st, function, local_namespace):
        var = st.edges[0].node.value
        value = st.edges[1].node
        if var in (function.variables_root | function.variables_free |
                   function.variables_local_closure):
            n = st.graph.apply("universe_setitem", local_namespace[var], value)
            n.edges[SEQ] = st.edges[SEQ]
            repl[st] = n
        elif var in function.variables_global:
            raise UnimplementedError("attempt to write to a global variable")
        elif var in function.variables_local:
            repl_seq[st] = st.edges[SEQ].node
        else:
            raise AssertionError(f"unclassified variable '{var}'")

    def resolve(self, functions):
        namespace = {}
        for function in functions:

            errs = function.variables_nonlocal - namespace.keys()
            for err in errs:
                raise SyntaxError(f"no binding for variable '{err}' found")

            for var in function.variables_root:
                st = function.variables_first_write[var]
                namespace[var] = st.graph.apply(
                    "make_handle",
                    st.graph.apply("typeof", st.edges[1]),
                )

            local_namespace = namespace.copy()
            for var in function.variables_local_closure:
                st = function.variables_first_write[var]
                local_namespace[var] = st.graph.apply(
                    "make_handle",
                    st.graph.apply("typeof", st.edges[1])
                )

            for block in function.blocks:
                if not block.used:
                    continue
                repl = {}
                repl_seq = {}

                # resolve all variable reads
                for var, items in block.variables_read.items():
                    for item in items:
                        self.resolve_read(
                            repl, repl_seq, item, function, local_namespace)

                # resolve write that need to stay
                for var, items in block.variables_written.items():
                    for item in items:
                        self.resolve_write(
                            repl, repl_seq, item, function, local_namespace)

                # make sure to "bake in" top level chain replacements
                # replacements inside the subgraphs are handled by replace
                for k in list(repl_seq):
                    n = repl_seq[k]
                    while n in repl_seq or n in repl:
                        if n in repl_seq:
                            n = repl_seq[n]
                        else:
                            n = repl[n]
                    repl_seq[k] = n

                for k in list(repl):
                    n = repl[k]
                    while n in repl:
                        n = repl[n]
                    repl[k] = n

                block.graph.replace(repl, repl_seq)

    def _create_function(self, block, node):
        function = Function(parent=block,
                            name=node.name,
                            location=self.make_location(node),
                            flags=self.recflags)
        function_block = function.initial_block

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
                # TODO If there are no parents (the initial function),
                # then evaluate in the global env
                if block is None:
                    raise MyiaSyntaxError("default value on the entry function")
                # XXX: This might not work correctly with our framework,
                # but we need to evaluate the default arguments in the parent
                # context for proper name resolution.
                dflt_node = self.process_node(block, dflt)
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
        tb = block.function.new_block("if_true", block, self.make_location(tn))
        fb = block.function.new_block("if_false", block, self.make_location(fn))
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

    def process_If(self, block, node):
        cond = self.process_node(block, node.test)
        true_block, false_block = self.make_condition_blocks(block, node.body, node.orelse)

        # TODO: figure out how to add a location here
        # (we would need the list of nodes that follow the if)
        after_block = block.function.new_block("if_after", block, None)
        after_block.used = False

        true_end = self.process_statements(true_block, node.body)
        if not true_end.graph.return_:
            after_block.used = True
            true_end.jump(after_block)

        false_end = self.process_statements(false_block, node.orelse)
        if not false_end.graph.return_:
            after_block.used = True
            false_end.jump(after_block)

        block.cond(cond, true_block, false_block)
        return after_block

    def process_Global(self, block, node):
        for name in node.names:
            if name in block.function.variables_local:
                # This is a python error
                raise SyntaxError(f"name '{name}' is assigned to before global declaration")
        block.function.variables_global.update(node.names)
        return block

    def process_Nonlocal(self, block, node):
        for name in node.names:
            if name in block.function.local_variables:
                # This is a python error
                raise SyntaxError(f"name '{name}' is assigned to before nonlocal declaration")
        block.function.variables_nonlocal.update(node.names)
        return block

    def process_Pass(self, block, node):
        return block

    def process_Return(self, block, node):
        block.returns(self.process_node(block, node.value))
        return block

    def process_While(self, block, node):
        header_block = block.function.new_block("while_header", block,
                                                self.make_location(node.test))
        body_block = block.function.new_block("while_body", header_block,
                                              self.make_location(node.body))
        # TODO: Same as If we need the list of nodes that follow
        # for the location
        after_block = block.function.new_block("while_after", header_block, None)

        block.jump(header_block)
        cond = self.process_node(header_block, node.test)
        header_block.cond(cond, body_block, after_block)

        after_body = self.process_statements(body_block, node.body)
        if not after_body.graph.return_:
            after_body.jump(header_block)

        return after_block
