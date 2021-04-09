import ast
import inspect
import operator

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
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.MatMult: operator.matmul,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.Gt: operator.gt,
    ast.LtE: operator.le,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: operator.contains,
    # ast.NotIn: # Not available in operator, special handling below
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

    def raises(self, exc):
        self.returns(self.apply("raise", exc))

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

        self.break_target = []
        self.continue_target = []

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
            assert st.is_apply()
            repl[ld] = st.edges[1].node
            repl_seq[ld] = ld.edges[SEQ].node
        else:
            raise AssertionError(f"unclassified variable '{var}'")  # pragma: nocover

    def resolve_write(self, repl, repl_seq, st, function, local_namespace):
        var = st.edges[0].node.value
        value = st.edges[1].node
        if var in (function.variables_root | function.variables_free |
                   function.variables_local_closure):
            n = st.graph.apply("universe_setitem", local_namespace[var], value)
            n.edges[SEQ] = st.edges[SEQ]
            repl[st] = n
        elif var in function.variables_global:
            raise NotImplementedError("attempt to write to a global variable")
        elif var in function.variables_local:
            repl_seq[st] = st.edges[SEQ].node
        else:
            raise AssertionError(f"unclassified variable '{var}'")  # pragma: nocover

    def resolve(self, functions):
        namespace = {}
        for function in functions:

            errs = function.variables_nonlocal - namespace.keys()
            for err in errs:
                # This should never show up in practice
                # since python will error out
                raise SyntaxError(f"no binding for variable '{err}' found")  # pragma: nocover

            for var in function.variables_root:
                st = function.variables_first_write[var]
                namespace[var] = st.graph.apply(
                    "make_handle",
                    st.graph.apply("typeof", st.edges[1].node),
                )

            local_namespace = namespace.copy()
            for var in function.variables_local_closure:
                st = function.variables_first_write[var]
                local_namespace[var] = st.graph.apply(
                    "make_handle",
                    st.graph.apply("typeof", st.edges[1].node)
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
                    while n in repl:  # pragma: nocover
                        assert False, "Please report the code that triggered this"
                        n = repl[n]
                    repl[k] = n

                block.graph.replace(repl, repl_seq)

    def _process_args(self, block, function_block, args):
        pargs = args.args
        nondefaults = [None] * (len(pargs) - len(args.defaults))
        defaults = nondefaults + args.defaults

        kwargs = args.kwonlyargs
        kwnondefaults = [None] * (len(kwargs) - len(args.kw_defaults))
        kwdefaults = kwnondefaults + args.kw_defaults

        defaults_name = []
        defaults_list = []

        for arg, dflt in zip(pargs + kwargs, defaults + kwdefaults):
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
                    raise MyiaSyntaxError("default value on the entry function", self.make_location(arg))
                # XXX: This might not work correctly with our framework,
                # but we need to evaluate the default arguments in the parent
                # context for proper name resolution.
                dflt_node = self.process_node(block, dflt)
                defaults_name.append(arg.arg)
                defaults_list.append(dflt_node)

        if args.vararg:
            arg = args.vararg
            vararg_node = Parameter(function_block.graph, name=arg.arg,
                                    location=self.make_location(arg))
            function_block.graph.parameters.append(vararg_node)
            function_block.write(arg.arg, vararg_node)
        else:
            vararg_node = None

        if args.kwarg:
            arg = args.kwarg
            kwarg_node = Parameter(function_block.graph, name=arg.arg,
                                   location=self.make_location(arg))
            function_block.graph.parameters.append(kwarg_node)
            function_block.write(arg.arg, kwarg_node)
        else:
            kwarg_node = None

        function_block.graph.vararg = vararg_node and vararg_node.name
        function_block.graph.kwarg = kwarg_node and kwarg_node.name
        function_block.graph.defaults = dict(zip(defaults_name,
                                                 defaults_list))
        function_block.graph.kwonly = len(args.kwonlyargs)

    def _create_function(self, block, node):
        function = Function(parent=block,
                            name=node.name,
                            location=self.make_location(node),
                            flags=self.recflags)
        function_block = function.initial_block

        self._process_args(block, function_block, node.args)

        after_block = self.process_statements(function_block, node.body)

        if after_block.graph.return_ is None:
            after_block.returns(Constant(None))

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

    def process_Attribute(self, block, node):
        value = self.process_node(block, node.value)
        return block.apply(getattr, value, Constant(node.attr))

    def process_BinOp(self, block, node):
        return block.apply(ast_map[type(node.op)],
                           self.process_node(block, node.left),
                           self.process_node(block, node.right))

    def _fold_bool(self, block, values, mode):
        first, *rest = values
        test = self.process_node(block, first)
        if rest:
            test = block.apply(operator.truth, test)
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
            raise AssertionError(f"Unknown BoolOp: {node.op}")  # pragma: nocover

    def process_Call(self, block, node):
        func = self.process_node(block, node.func)

        groups = []
        current = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                groups.append(current)
                groups.append(self.process_node(block, arg.value))
                current = []
            else:
                current.append(self.process_node(block, arg))
        if current or not groups:
            groups.append(current)

        if node.keywords:
            for k in node.keywords:
                if k.arg is None:
                    groups.append(self.process_node(block, k.value))
            keywords = [k for k in node.keywords if k.arg is not None]
            kwlist = list(zip((k.arg for k in keywords), (self.process_node(block, k.value) for k in keywords)))
            dlist = []
            for kw in kwlist:
                dlist.extend(kw)
            groups.append(block.apply("make_dict", *dlist))

        if len(groups) == 1:
            (args,) = groups
            return block.apply(func, *args)
        else:
            args = []
            for group in groups:
                if isinstance(group, list):
                    args.append(block.apply("make_tuple", *group))
                else:
                    args.append(group)
            return block.apply("apply", func, *args)

    def process_Compare(self, block, node):
        if len(node.ops) == 1:
            left = self.process_node(block, node.left)
            right = self.process_node(block, node.comparators[0])
            if type(node.ops[0]) is ast.NotIn:
                # NotIn doesn't have an operator mapping
                return block.apply(operator.not_,
                                   block.apply(operator.contains,
                                               left, right))
            else:
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

    def process_Dict(self, block, node):
        # we need to process k1, v1, k2, v2, ...
        # to respect python evaluation order
        dlist = []
        for k, v in zip(node.keys, node.values):
            dlist.append(self.process_node(block, k))
            dlist.append(self.process_node(block, v))

        return block.apply("make_dict", *dlist)

    def process_ExtSlice(self, block, node):
        # This node is removed in 3.9+
        slices = [self.process_node(block, dim) for dim in node.dims]
        return block.apply("make_tuple", *slices)

    def process_IfExp(self, block, node):
        cond = self.process_node(block, node.test)
        cond = block.apply(operator.truth, cond)
        tb, fb = self.make_condition_blocks(block, node.body, node.orelse)

        tn = self.process_node(tb, node.body)
        fn = self.process_node(fb, node.orelse)

        tb.returns(tn)
        fb.returns(fn)

        switch = block.apply("user_switch", cond, tb.graph, fb.graph)
        return block.apply(switch)

    def process_Index(self, block, node):
        return self.process_node(block, node.value)

    def process_Lambda(self, block, node):
        function = Function(parent=block,
                            name="lambda",
                            location=self.make_location(node),
                            flags=self.recflags)
        function_block = function.initial_block

        self._process_args(block, function_block, node.args)

        function_block.returns(self.process_node(function_block, node.body))
        return Constant(function_block.graph)

    def process_List(self, block, node):
        elts = [self.process_node(block, e) for e in node.elts]
        return block.apply("make_list", *elts)

    def process_Name(self, block, node):
        assert isinstance(node.ctx, ast.Load)
        return block.read(node.id)

    def process_NameConstant(self, block, node):
        # removed in python 3.8
        return Constant(node.value)  # pragma: nocover

    def process_Slice(self, block, node):
        if node.lower is None:
            lower = Constant(None)
        else:
            lower = self.process_node(block, node.lower)
        if node.upper is None:
            upper = Constant(None)
        else:
            upper = self.process_node(block, node.upper)
        if node.step is None:
            step = Constant(None)
        else:
            step = self.process_node(block, node.step)
        return block.apply("slice", lower, upper, step)

    def process_Subscript(self, block, node):
        value = self.process_node(block, node.value)
        slice = self.process_node(block, node.slice)
        return block.apply(operator.getitem, value, slice)

    def process_Tuple(self, block, node):
        elts = [self.process_node(block, e) for e in node.elts]
        if len(elts) == 0:
            return Constant(())
        else:
            return block.apply("make_tuple", *elts)

    def process_UnaryOp(self, block, node):
        val = self.process_node(block, node.operand)
        return block.apply(ast_map[type(node.op)], val)

    # statements (returns a block)

    def process_statements(self, starting_block, nodes):
        block = starting_block
        for node in nodes:
            block = self.process_node(block, node)
        return block

    def _assign(self, block, targ, idx, val):
        if isinstance(targ, ast.Name):
            # x = val
            if idx is not None:
                val = block.apply(operator.getitem, val, idx)
            st = block.write(targ.id, val)

        elif isinstance(targ, (ast.Tuple, ast.List)):
            # x, y = val
            if idx is not None:
                val = block.apply(operator.getitem, val, idx)
            for i, elt in enumerate(targ.elts):
                self._assign(block, elt, i, val)

        elif isinstance(targ, ast.Starred):
            if idx is None:
                # this should not show up since python will catch it
                raise SyntaxError("starred assignement target must be in a list or tuple")  # pragma: nocover
            else:
                raise NotImplementedError("starred assignement")

        else:
            raise NotImplementedError(targ)

    def process_AnnAssign(self, block, node):
        val = self.process_node(block, node.value)
        val.add_annotation(self._eval_ast_node(node.annotation))
        self._assign(block, node.target, None, val)
        return block

    def process_Assert(self, block, node):
        cond = self.process_node(block, node.test)
        cond = block.apply(operator.truth, cond)
        msg = self.process_node(block, node.msg) if node.msg else Constant("Assertion failed")

        true_block, false_block = self.make_condition_blocks(block, None, None)
        block.cond(cond, true_block, false_block)
        false_block.raises(false_block.apply("exception", msg))
        return true_block

    def process_Assign(self, block, node):
        val = self.process_node(block, node.value)
        for targ in node.targets:
            self._assign(block, targ, None, val)

        return block

    def process_Break(self, block, node):
        if len(block.function.break_target) == 0:
            # python should catch this
            raise SyntaxError("'break' outside loop")  # pragma: nocover
        block.jump(block.function.break_target[-1])
        return block

    def process_Continue(self, block, node):
        target = block.function.continue_target
        if len(target) == 0:
            # python should catch this
            raise SyntaxError("'continue' not properly in loop")  # pragma: nocover
        target = target[-1]
        block.jump(target[0], *target[1])
        return block

    def process_Expr(self, block, node):
        self.process_node(block, node.value)
        return block

    def process_For(self, block, node):
        init = block.apply("python_iter", self.process_node(block, node.iter))

        header_block = block.function.new_block("for_header", block, None)
        it = header_block.graph.add_parameter('it')
        cond = header_block.apply("python_hasnext", it)

        body_block = block.function.new_block("for_body", header_block,
                                              self.make_location(node.body))
        app = body_block.apply("python_next", it)
        val = body_block.apply(operator.getitem, app, 0)
        self._assign(body_block, node.target, None, val)
        it2 = body_block.apply(operator.getitem, app, 1)

        else_block = block.function.new_block("for_else", header_block,
                                              self.make_location(node.orelse))
        after_block = block.function.new_block("for_after", block, None)

        block.jump(header_block, init)
        header_block.cond(cond, body_block, else_block)

        block.function.break_target.append(after_block)
        block.function.continue_target.append((header_block, (it2,)))

        after_body_block = self.process_statements(body_block, node.body)
        if not after_body_block.graph.return_:
            after_body_block.jump(header_block, it2)

        block.function.break_target.pop(-1)
        block.function.continue_target.pop(-1)

        after_else_block = self.process_statements(else_block, node.orelse)
        if not after_else_block.graph.return_:
            after_else_block.jump(after_block)

        return after_block

    def process_FunctionDef(self, block, node):
        fn_block = self._create_function(block, node)
        block.write(node.name, fn_block.graph)
        return block

    def process_If(self, block, node):
        cond = self.process_node(block, node.test)
        cond = block.apply(operator.truth, cond)
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
                raise SyntaxError(f"name '{name}' is assigned to before global declaration")  # pragma: nocover
        block.function.variables_global.update(node.names)
        return block

    def process_Nonlocal(self, block, node):
        for name in node.names:
            if name in block.function.variables_local:
                # This is a python error
                raise SyntaxError(f"name '{name}' is assigned to before nonlocal declaration")  # pragma: nocover
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
        else_block = block.function.new_block("while_else", header_block,
                                              self.make_location(node.orelse))
        # TODO: Same as If we need the list of nodes that follow
        # for the location
        after_block = block.function.new_block("while_after", header_block, None)

        block.jump(header_block)
        cond = self.process_node(header_block, node.test)
        header_block.cond(cond, body_block, else_block)

        block.function.break_target.append(after_block)
        block.function.continue_target.append((header_block, ()))

        after_body = self.process_statements(body_block, node.body)
        if not after_body.graph.return_:
            after_body.jump(header_block)

        block.function.break_target.pop(-1)
        block.function.continue_target.pop(-1)

        after_else = self.process_statements(else_block, node.orelse)
        if not after_else.graph.return_:
            after_else.jump(after_block)

        return after_block
