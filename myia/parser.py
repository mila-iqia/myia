"""Parser module to transform python code into myia IR."""

import ast
import inspect
import operator
import textwrap
from typing import NamedTuple

from . import basics
from .ir import Apply, Constant, Graph, Parameter
from .ir.node import SEQ
from .utils import ModuleNamespace, Named
from .utils.info import about, debug_inherit, get_debug

STORE = Named("STORE")
LOAD = Named("LOAD")


class Location(NamedTuple):
    """A location in source code.

    Attributes:
        filename: The filename.
        line: The initial line number.
        column: The initial column number.
        line_end: The final line number
        column_end: The final column number
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
    """Parse a python function and return the myia graph.

    This takes a python function object and will extract and parse its
    source code.
    """
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
        graph.add_debug(name=name)
    _parse_cache[func] = graph
    return graph


class _CondJump(NamedTuple):
    from_: "Block"
    true: "Block"
    false: "Block"
    jump: "Apply"
    vars: dict
    done: set


class _Jump(NamedTuple):
    from_: "Block"
    target: "Block"
    jump: "Apply"
    vars: dict
    done: set


class Block:
    """This is used to represent a basic block in a python function."""

    def __init__(self, function, parent_graph, flags={}):
        self.function = function

        self.graph = Graph(parent_graph)
        self.graph.set_flags(reference=True)
        self.graph.flags.update(flags)

        self.used = True
        self.preds = []
        self.phis = dict()

        self.last_apply = None

        self.variables_written = {}
        self.variables_read = {}

    def apply(self, *inputs):
        """Create an apply and link it in the sequence chain."""
        res = self.graph.apply(*inputs)
        self.link_seq(res)
        return res

    def read(self, varnum):
        """Add a read operation of the specified name.

        This will return a placeholder 'load' operation that will be
        resolved later when we have more information about where the
        variable is read from.
        """
        ld = self.apply(LOAD, varnum)
        st = self.variables_written.get(varnum, [None])[-1]
        ld.add_edge("prevwrite", st)
        self.variables_read.setdefault(varnum, []).append(ld)
        if varnum not in self.function.variables_local:
            self.function.variables_free.add(varnum)
        return ld

    def write(self, varnum, node):
        """Add a write operation of the specified value to the specified name.

        This will add a placeholder 'store' operation in the sequence
        chain that will be resolved later.
        """
        st = self.apply(STORE, varnum, node)
        self.variables_written.setdefault(varnum, []).append(st)
        self.function.variables_local.add(varnum)
        self.function.variables_first_write.setdefault(varnum, st)

    def link_seq(self, node):
        """Append a node in the sequence chain."""
        if self.last_apply is not None:
            node.add_edge(SEQ, self.last_apply)
        self.last_apply = node
        return node

    def cond(self, cond, true, false):
        """End the block with a conditional jump to one of two blocks."""
        assert self.graph.return_ is None
        switch = self.apply(basics.user_switch, cond, true.graph, false.graph)
        jump = self.apply(switch)
        tag = _CondJump(
            self, true, false, jump, self.variables_written.copy(), set()
        )
        true.preds.append(tag)
        false.preds.append(tag)
        self.returns(jump)


    def jump(self, target, *args):
        """End the block in a jump to another block, with optional arguments."""
        assert self.graph.return_ is None
        jump = self.apply(target.graph, *args)
        target.preds.append(
            _Jump(self, target, jump, self.variables_written.copy(), set())
        )
        self.returns(jump)

    def raises(self, exc):
        """End the block with an exception."""
        self.returns(self.apply(basics.raise_, exc))

    def returns(self, value):
        """End the block with an arbitrary expression."""
        assert self.graph.return_ is None
        self.graph.output = value
        self.link_seq(self.graph.return_)


class Function:
    """Represents a python fucntion, mostly for namespace purposes."""

    def __init__(self, parent, flags={}):
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

        self.initial_block = Block(self, parent_graph, flags)
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

    def new_block(self, parent):
        """Create a new block in this function.

        The parent must be a block in which all the closed over
        variables are defined in either it or its parents.
        """
        block = Block(self, parent.graph, self.flags)
        self.blocks.append(block)
        return block


class Parser:
    """Utility class for the parsing state, use `parse` instead."""

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
        self.finalizers = {}

    def _eval_ast_node(self, node):
        text = ast.get_source_segment(self.src, node)
        # XXX: This needs the locals at the point of the annotation
        # to match "old" python behaviour.

        # We currently act as if "from __future__ import annotations" is
        # in effect, which means that we only need to care about globals.
        return eval(text, self.function.__globals__)

    def make_location(self, node):
        """Make a `Location` for a node or list of nodes."""
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
        """Perform the parsing of the top-level function and subfunctions."""
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
        """Return a list of all the subfunctions in a function in preorder."""
        functions = [main_function]
        todo = list(main_function.children)
        while todo:
            fn = todo.pop()
            functions.append(fn)
            todo.extend(fn.children)
        return functions

    def analyze(self, functions):
        """Analyze the functions to prepare for load/store resolution.

        This will traverse all the blocks in all the functions and
        collect information about variables definitions and closures.
        This will modify the Block and Function objects to store that
        information.
        """
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
                raise UnboundLocalError(
                    f"local variable '{err}' is referenced before assignment"
                )

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
                    if (
                        var in function.variables_local
                        and reads[0].edges["prevwrite"].node is None
                    ):
                        block.phis[var] = None
                        function.variables_local_closure.add(var)

    def resolve_read(
        self, repl, repl_seq, ld, function, block, local_namespace
    ):
        """Resolve a 'load' operation with pre-collected information."""
        var = ld.edges[0].node.value
        st = ld.edges["prevwrite"].node
        if var in (function.variables_root | function.variables_free):
            if var not in local_namespace:
                n = ld.graph.apply(basics.resolve, self.global_namespace, var)
            else:
                n = ld.graph.apply(
                    basics.global_universe_getitem, local_namespace[var]
                )
            if SEQ in ld.edges:
                n.edges[SEQ] = ld.edges[SEQ]
            repl[ld] = n
        elif var in function.variables_local_closure:
            if st is None:
                repl[ld] = local_namespace[var]
            else:
                assert st.is_apply()
                repl[ld] = st.edges[1].node
            if SEQ in ld.edges:
                repl_seq[ld] = ld.edges[SEQ].node
            else:
                repl_seq[ld] = None
        elif var in function.variables_global:
            n = ld.graph.apply(basics.resolve, self.global_namespace, var)
            if SEQ in ld.edges:
                n.edges[SEQ] = ld.edges[SEQ]
            repl[ld] = n
        elif var in function.variables_local:
            assert st.is_apply()
            repl[ld] = st.edges[1].node
            # There should always be at least one store before this load
            assert SEQ in ld.edges
            repl_seq[ld] = ld.edges[SEQ].node
        else:
            raise AssertionError(
                f"unclassified variable '{var}'"
            )  # pragma: no cover

    def resolve_write(
        self, repl, repl_seq, st, function, block, local_namespace
    ):
        """Resolve a 'store' operation with pre-collected information."""
        var = st.edges[0].node.value
        value = st.edges[1].node
        if var in (function.variables_root | function.variables_free):
            n = st.graph.apply(
                basics.global_universe_setitem, local_namespace[var], value
            )
            if SEQ in st.edges:
                n.edges[SEQ] = st.edges[SEQ]
            repl[st] = n
        elif var in function.variables_global:
            raise NotImplementedError("attempt to write to a global variable")
        elif var in function.variables_local:
            if get_debug():
                st.edges[1].node.debug.name = st.edges[0].node.value
            if SEQ in st.edges:
                repl_seq[st] = st.edges[SEQ].node
            else:
                repl_seq[st] = None
        else:
            raise AssertionError(
                f"unclassified variable '{var}'"
            )  # pragma: no cover

    def process_phi(self, block, phi):
        """Add the phi argument to the block.

        This also fixes call sites to pass in the value. This may act
        recursively in case the parent block doesn't have the value as
        a local value.
        """
        # Skip processing if it's already there
        val = block.phis.get(phi, None)
        if val is not None:
            return val

        # Add an argument for the value of the phi
        block.phis[phi] = block.graph.add_parameter("phi_" + phi)

        # Add the arguments in all the pred calls
        for pred in block.preds:
            assert phi in block.function.variables_local_closure
            if phi not in pred.vars:
                val = self.process_phi(pred.from_, phi)
            else:
                val = pred.vars[phi][-1].edges[1].node
            if phi not in pred.done:
                pred.jump.append_input(val)
                pred.done.add(phi)
            if isinstance(pred, _CondJump):
                if pred.true.phis.get(phi, None) is None:
                    p = pred.true.graph.add_parameter("phi_" + phi)
                    pred.true.phis[phi] = p
                if pred.false.phis.get(phi, None) is None:
                    p = pred.false.graph.add_parameter("phi_" + phi)
                    pred.false.phis[phi] = p
        return block.phis[phi]

    def resolve(self, functions):
        """Resolve all the 'load' and 'store' operations.

        This assumes that `Parser.analyze` has been called first and
        will use the information it collects.
        """
        namespace = {}
        for function in functions:

            errs = function.variables_nonlocal - namespace.keys()
            for err in errs:
                # This should never show up in practice
                # since python will error out
                raise SyntaxError(
                    f"no binding for variable '{err}' found"
                )  # pragma: no cover

            for var in function.variables_root:
                st = function.variables_first_write[var]
                t = st.graph.apply(type, st.edges[1].node)
                with debug_inherit(name=st.edges[0].node.value):
                    namespace[var] = st.graph.apply(basics.make_handle, t)

            for block in function.blocks:
                if not block.used:
                    continue
                repl = {}
                repl_seq = {}
                local_namespace = namespace.copy()

                for phi in list(block.phis.keys()):
                    local_namespace[phi] = self.process_phi(block, phi)

                # resolve all variable reads
                for var, items in block.variables_read.items():
                    for item in items:
                        self.resolve_read(
                            repl,
                            repl_seq,
                            item,
                            function,
                            block,
                            local_namespace,
                        )

                # resolve write that need to stay
                for var, items in block.variables_written.items():
                    for item in items:
                        self.resolve_write(
                            repl,
                            repl_seq,
                            item,
                            function,
                            block,
                            local_namespace,
                        )

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
                    while n in repl:  # pragma: no cover
                        assert (
                            False
                        ), "Please report the code that triggered this"
                        n = repl[n]
                    repl[k] = n

                block.graph.replace(repl, repl_seq)

    def _process_args(self, block, function_block, args):
        """Process argument definition.

        This can be used for any argument list.
        """
        pargs = args.args
        nondefaults = [None] * (len(pargs) - len(args.defaults))
        defaults = nondefaults + args.defaults

        kwargs = args.kwonlyargs
        kwnondefaults = [None] * (len(kwargs) - len(args.kw_defaults))
        kwdefaults = kwnondefaults + args.kw_defaults

        defaults_name = []
        defaults_list = []

        # We don't handle this for now.
        assert args.posonlyargs == []

        for arg, dflt in zip(pargs + kwargs, defaults + kwdefaults):
            with debug_inherit(name=arg.arg, location=self.make_location(arg)):
                param_node = Parameter(function_block.graph, name=arg.arg)

            if arg.annotation:
                param_node.annotation = self._eval_ast_node(arg.annotation)

            function_block.graph.parameters.append(param_node)
            function_block.write(arg.arg, param_node)
            if dflt:
                # TODO If there are no parents (the initial function),
                # then evaluate in the global env
                if block is None:
                    raise MyiaSyntaxError(
                        "default value on the entry function",
                        self.make_location(arg),
                    )
                # XXX: This might not work correctly with our framework,
                # but we need to evaluate the default arguments in the parent
                # context for proper name resolution.
                dflt_node = self.process_node(block, dflt)
                defaults_name.append(arg.arg)
                defaults_list.append(dflt_node)

        if args.vararg:
            arg = args.vararg
            with debug_inherit(name=arg.arg, location=self.make_location(arg)):
                vararg_node = Parameter(function_block.graph, name=arg.arg)
            function_block.graph.parameters.append(vararg_node)
            function_block.write(arg.arg, vararg_node)
        else:
            vararg_node = None

        if args.kwarg:
            arg = args.kwarg
            with debug_inherit(name=arg.arg, location=self.make_location(arg)):
                kwarg_node = Parameter(function_block.graph, name=arg.arg)
            function_block.graph.parameters.append(kwarg_node)
            function_block.write(arg.arg, kwarg_node)
        else:
            kwarg_node = None

        function_block.graph.varargs = vararg_node
        function_block.graph.kwargs = kwarg_node
        function_block.graph.defaults = dict(zip(defaults_name, defaults_list))
        function_block.graph.kwonly = len(args.kwonlyargs)

    def _create_function(self, block, node):
        with debug_inherit(name=node.name, location=self.make_location(node)):
            function = Function(
                parent=block,
                flags=self.recflags,
            )
        function_block = function.initial_block

        self._process_args(block, function_block, node.args)

        after_block = self.process_statements(function_block, node.body)

        if after_block.graph.return_ is None:
            after_block.returns(Constant(None))

        if node.returns:
            function_block.graph.return_.annotation = self._eval_ast_node(
                node.returns
            )

        return function_block

    def make_condition_blocks(self, block, tn, fn):
        """Make true/false branch blocks."""
        with about(
            block.graph, relation="if_true", location=self.make_location(tn)
        ):
            tb = block.function.new_block(block)
        with about(
            block.graph, relation="if_false", location=self.make_location(fn)
        ):
            fb = block.function.new_block(block)
        return tb, fb

    # expressions (returns a value)

    def process_node(self, block, node):
        """Process an AST node in the current block."""
        method_name = f"_process_{node.__class__.__name__}"
        method = getattr(self, method_name, None)
        if method:
            return method(block, node)
        else:
            raise MyiaSyntaxError(
                f"{node.__class__.__name__} not supported",
                self.make_location(node),
            )

    def _process_Attribute(self, block, node):
        value = self.process_node(block, node.value)
        return block.apply(getattr, value, Constant(node.attr))

    def _process_BinOp(self, block, node):
        return block.apply(
            ast_map[type(node.op)],
            self.process_node(block, node.left),
            self.process_node(block, node.right),
        )

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
            switch = block.apply(basics.switch, test, tb.graph, fb.graph)
            jmp = block.apply(switch)
            tag = _CondJump(
                block, tb, fb, jmp, block.variables_written.copy(), set()
            )
            tb.preds.append(tag)
            fb.preds.append(tag)
            return jmp
        else:
            return test

    def _process_BoolOp(self, block, node):
        if isinstance(node.op, ast.And):
            return self._fold_bool(block, node.values, "and")
        elif isinstance(node.op, ast.Or):
            return self._fold_bool(block, node.values, "or")
        else:
            raise AssertionError(
                f"Unknown BoolOp: {node.op}"
            )  # pragma: no cover

    def _process_Call(self, block, node):
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
            kwlist = list(
                zip(
                    (k.arg for k in keywords),
                    (self.process_node(block, k.value) for k in keywords),
                )
            )
            dlist = []
            for kw in kwlist:
                dlist.extend(kw)
            groups.append(block.apply(basics.make_dict, *dlist))

        if len(groups) == 1:
            (args,) = groups
            return block.apply(func, *args)
        else:
            args = []
            for group in groups:
                if isinstance(group, list):
                    args.append(block.apply(basics.make_tuple, *group))
                else:
                    args.append(group)
            return block.apply(basics.apply, func, *args)

    def _process_Compare(self, block, node):
        if len(node.ops) == 1:
            left = self.process_node(block, node.left)
            right = self.process_node(block, node.comparators[0])
            if type(node.ops[0]) is ast.NotIn:
                # NotIn doesn't have an operator mapping
                return block.apply(
                    operator.not_, block.apply(operator.contains, left, right)
                )
            else:
                return block.apply(ast_map[type(node.ops[0])], left, right)
        else:
            cur = node.left
            rest = node.comparators
            ops = node.ops
            values = []
            while ops:
                # TODO: fix up source locations
                values.append(
                    ast.Compare(ops=[ops[0]], left=cur, comparators=[rest[0]])
                )
                cur, rest = rest[0], rest[1:]
                ops = ops[1:]
            return self._fold_bool(block, values, "and")

    def _process_Constant(self, block, node):
        return Constant(node.value)

    def _process_Dict(self, block, node):
        # we need to process k1, v1, k2, v2, ...
        # to respect python evaluation order
        dlist = []
        for k, v in zip(node.keys, node.values):
            dlist.append(self.process_node(block, k))
            dlist.append(self.process_node(block, v))

        return block.apply(basics.make_dict, *dlist)

    def _process_ExtSlice(self, block, node):
        # This node is removed in 3.9+
        slices = [self.process_node(block, dim) for dim in node.dims]
        return block.apply(basics.make_tuple, *slices)

    def _process_IfExp(self, block, node):
        cond = self.process_node(block, node.test)
        cond = block.apply(operator.truth, cond)
        tb, fb = self.make_condition_blocks(block, node.body, node.orelse)

        tn = self.process_node(tb, node.body)
        fn = self.process_node(fb, node.orelse)

        tb.returns(tn)
        fb.returns(fn)

        switch = block.apply(basics.user_switch, cond, tb.graph, fb.graph)
        jmp = block.apply(switch)
        tag = _CondJump(
            block, tb, fb, jmp, block.variables_written.copy(), set()
        )
        tb.preds.append(tag)
        fb.preds.append(tag)
        return jmp

    def _process_Index(self, block, node):
        return self.process_node(block, node.value)

    def _process_Lambda(self, block, node):
        with debug_inherit(name="lambda", location=self.make_location(node)):
            function = Function(
                parent=block,
                flags=self.recflags,
            )
        function_block = function.initial_block

        self._process_args(block, function_block, node.args)

        function_block.returns(self.process_node(function_block, node.body))
        return Constant(function_block.graph)

    def _process_List(self, block, node):
        elts = [self.process_node(block, e) for e in node.elts]
        return block.apply(basics.make_list, *elts)

    def _process_Name(self, block, node):
        assert isinstance(node.ctx, ast.Load)
        return block.read(node.id)

    def _process_Slice(self, block, node):
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
        return block.apply(slice, lower, upper, step)

    def _process_Subscript(self, block, node):
        value = self.process_node(block, node.value)
        slice = self.process_node(block, node.slice)
        return block.apply(operator.getitem, value, slice)

    def _process_Tuple(self, block, node):
        elts = [self.process_node(block, e) for e in node.elts]
        if len(elts) == 0:
            return Constant(())
        else:
            return block.apply(basics.make_tuple, *elts)

    def _process_UnaryOp(self, block, node):
        val = self.process_node(block, node.operand)
        return block.apply(ast_map[type(node.op)], val)

    # statements (returns a block)

    def process_statements(self, starting_block, nodes):
        """Process a list of statements.

        This will return the active block at the end of the list which
        may be different from the `starting_block`.
        """
        block = starting_block
        for node in nodes:
            block = self.process_node(block, node)
        return block

    def _assign(self, block, targ, idx, val):
        if isinstance(targ, ast.Name):
            # x = val
            if idx is not None:
                val = block.apply(operator.getitem, val, idx)
            block.write(targ.id, val)

        elif isinstance(targ, (ast.Tuple, ast.List)):
            # x, y = val
            if idx is not None:
                val = block.apply(operator.getitem, val, idx)
            for i, elt in enumerate(targ.elts):
                self._assign(block, elt, i, val)

        elif isinstance(targ, ast.Starred):
            if idx is None:
                # this should not show up since python will catch it
                raise SyntaxError(
                    "starred assignement target must be in a list or tuple"
                )  # pragma: no cover
            else:
                raise NotImplementedError("starred assignement")

        else:
            raise NotImplementedError(targ)

    def _process_AnnAssign(self, block, node):
        val = self.process_node(block, node.value)
        val.annotation = self._eval_ast_node(node.annotation)
        self._assign(block, node.target, None, val)
        return block

    def _process_Assert(self, block, node):
        cond = self.process_node(block, node.test)
        cond = block.apply(operator.truth, cond)
        msg = (
            self.process_node(block, node.msg)
            if node.msg
            else Constant("Assertion failed")
        )

        true_block, false_block = self.make_condition_blocks(block, None, None)
        block.cond(cond, true_block, false_block)
        false_block.raises(false_block.apply(Exception, msg))
        return true_block

    def _process_Assign(self, block, node):
        val = self.process_node(block, node.value)
        for targ in node.targets:
            self._assign(block, targ, None, val)

        return block

    def _process_Break(self, block, node):
        if len(block.function.break_target) == 0:
            # python should catch this
            raise SyntaxError("'break' outside loop")  # pragma: no cover
        block.jump(block.function.break_target[-1])
        return block

    def _process_Continue(self, block, node):
        target = block.function.continue_target
        if len(target) == 0:
            # python should catch this
            raise SyntaxError(
                "'continue' not properly in loop"
            )  # pragma: no cover
        target = target[-1]
        block.jump(target[0], *target[1])
        return block

    def _process_Expr(self, block, node):
        self.process_node(block, node.value)
        return block

    def _process_For(self, block, node):
        init = block.apply(
            basics.myia_iter, self.process_node(block, node.iter)
        )

        with about(block.graph, relation="for"):
            header_block = block.function.new_block(block)
        it = header_block.graph.add_parameter("it")
        cond = header_block.apply(basics.myia_hasnext, it)

        with about(
            header_block.graph,
            relation="body",
            location=self.make_location(node.body),
        ):
            body_block = block.function.new_block(
                header_block,
            )
        app = body_block.apply(basics.myia_next, it)
        val = body_block.apply(operator.getitem, app, 0)
        self._assign(body_block, node.target, None, val)
        it2 = body_block.apply(operator.getitem, app, 1)

        with about(
            header_block.graph,
            relation="else",
            location=self.make_location(node.orelse),
        ):
            else_block = block.function.new_block(
                header_block,
            )
        with about(block.graph, relation="for_after"):
            after_block = block.function.new_block(block)

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

    def _process_FunctionDef(self, block, node):
        fn_block = self._create_function(block, node)
        block.write(node.name, fn_block.graph)
        return block

    def _process_If(self, block, node):
        cond = self.process_node(block, node.test)
        cond = block.apply(operator.truth, cond)
        true_block, false_block = self.make_condition_blocks(
            block, node.body, node.orelse
        )

        # TODO: figure out how to add a location here
        # (we would need the list of nodes that follow the if)
        with about(block.graph, relation="if_after"):
            after_block = block.function.new_block(block)
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

    def _process_Global(self, block, node):
        for name in node.names:
            if name in block.function.variables_local:
                # This is a python error
                raise SyntaxError(
                    f"name '{name}' is assigned to before global declaration"
                )  # pragma: no cover
        block.function.variables_global.update(node.names)
        return block

    def _process_Nonlocal(self, block, node):
        for name in node.names:
            if name in block.function.variables_local:
                # This is a python error
                raise SyntaxError(
                    f"name '{name}' is assigned to before nonlocal declaration"
                )  # pragma: no cover
        block.function.variables_nonlocal.update(node.names)
        return block

    def _process_Pass(self, block, node):
        return block

    def _process_Return(self, block, node):
        block.returns(self.process_node(block, node.value))
        return block

    def _process_While(self, block, node):
        with about(
            block.graph,
            relation="while",
            location=self.make_location(node.test),
        ):
            header_block = block.function.new_block(
                block,
            )
        with about(
            header_block.graph,
            relation="body",
            location=self.make_location(node.body),
        ):
            body_block = block.function.new_block(
                header_block,
            )
        with about(
            header_block.graph,
            relation="else",
            location=self.make_location(node.orelse),
        ):
            else_block = block.function.new_block(
                header_block,
            )
        # TODO: Same as If we need the list of nodes that follow
        # for the location
        with about(block.graph, relation="while_after"):
            after_block = block.function.new_block(header_block)

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
