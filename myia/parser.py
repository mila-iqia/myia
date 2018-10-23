"""Parse a Python AST into the Myia graph-based ANF IR.

Graph construction proceeds very similarly to the way that FIRM constructs its
SSA graph [1]_. The correspondence between functional representations and SSA
is discussed in detail in [2]_. The main takeaway is that basic blocks
correspond to functions, and jumping from one block to another is done by a
tail call. Phi nodes become formal parameters of the function. The inputs to a
phi node become actual parameters (arguments) passed at the call sites in the
predecessor blocks.

Note that variable names in this module closely follow the ones used in [1]_ in
order to make it easy to compare the two algorithms.

The parsing of Python functions is handled as follows:

Parser
  There is one parser per user-defined function. A parser is responsible for
  managing the parsing process.

Block
  A single basic block exists for each block of code in a user-defined
  function. Basic blocks are responsible for resolving variable names locally
  during parsing and converting Python's imperative control flow structures to
  a functional representation.

.. [1] Braun, M., Buchwald, S. and Zwinkau, A., 2011. Firm-A graph-based
   intermediate representation. KIT, Fakultät für Informatik.
.. [2] Appel, A.W., 1998. SSA is functional programming. ACM SIGPLAN Notices,
   33(4), pp.17-20.

"""

import ast
import asttokens
import inspect
import textwrap
from types import FunctionType
from typing import Dict, List, NamedTuple, Optional, Tuple, \
    overload as pyoverload

from .info import About, DebugInherit, NamedDebugInfo
from .ir import ANFNode, Apply, Constant, Graph, Parameter
from .prim import ops as primops
from .utils import ModuleNamespace, ClosureNamespace


_parse_cache = {}


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
    node: ast.AST


class MyiaSyntaxError(Exception):
    def __init__(self, msg, loc):
        super().__init__(msg)
        self.loc = loc


operations_ns = ModuleNamespace('myia.operations')
builtins_ns = ModuleNamespace('builtins')


ast_map = {
    ast.Add: (operations_ns, 'add'),
    ast.Sub: (operations_ns, 'sub'),
    ast.Mult: (operations_ns, 'mul'),
    ast.Div: (operations_ns, 'truediv'),
    ast.FloorDiv: (operations_ns, 'floordiv'),
    ast.Mod: (operations_ns, 'mod'),
    ast.Pow: (operations_ns, 'pow'),
    ast.MatMult: (operations_ns, 'matmul'),
    ast.LShift: (operations_ns, 'lshift'),
    ast.RShift: (operations_ns, 'rshift'),
    ast.BitAnd: (operations_ns, 'and_'),
    ast.BitOr: (operations_ns, 'or_'),
    ast.BitXor: (operations_ns, 'xor'),
    ast.UAdd: (operations_ns, 'pos'),
    ast.USub: (operations_ns, 'neg'),
    ast.Invert: (operations_ns, 'invert'),
    ast.Not: (operations_ns, 'not_'),
    ast.Eq: (operations_ns, 'eq'),
    ast.NotEq: (operations_ns, 'ne'),
    ast.Lt: (operations_ns, 'lt'),
    ast.Gt: (operations_ns, 'gt'),
    ast.LtE: (operations_ns, 'le'),
    ast.GtE: (operations_ns, 'ge'),
    ast.Is: (operations_ns, 'is_'),
    ast.IsNot: (operations_ns, 'is_not'),
    ast.In: (operations_ns, 'contains'),
    # ast.NotIn: operator.???,  # Not in operator module
}


def _fresh(node):
    """If node is a constant, return a copy of it."""
    if node.is_constant():
        return Constant(node.value)
    else:
        return node


def parse(func):
    """Parse a function into a Myia graph.

    The result of the parsing is cached: multiple calls to parse on the same
    function will return the same graph. It should therefore be cloned prior
    to manipulation.
    """
    if func in _parse_cache:
        return _parse_cache[func]
    parser = Parser(func)
    graph = parser.parse()
    graph.flags.update(getattr(func, '_myia_flags', {}))
    _parse_cache[func] = graph
    return graph


class Parser:
    """Parser for a function.

    This class handles the parsing of a single user-defined function.

    References to global variables, or nonlocal variables, are converted to
    calls to the `resolve` primitive (`resolve(ns, name)`).

    Attributes:
        function: The function to transform.
        filename: The name of the file in which the function is defined.
        global_namespace: The Namespace in which to resolve the function's
            global variables. It will be embedded in the graph for every
            global variable to resolve.
        closure_namespace: The Namespace in which to resolve the function's
            nonlocal variables. It will be embedded in the graph for every
            nonlocal variable to resolve.
        graph: The graph of the function being parsed.

    """

    def __init__(self, function: FunctionType) -> None:
        """Construct a parser."""
        self.function = function
        _, self.line_offset = inspect.getsourcelines(function)
        self.filename: str = inspect.getfile(function)
        # This is used to resolve the function's globals.
        self.global_namespace = ModuleNamespace(function.__module__)
        # This is used to resolve the function's nonlocals.
        self.closure_namespace = ClosureNamespace(self.function)
        # Will be set later
        self.graph = None

    def make_location(self, node) -> Location:
        """Create a Location from an AST node."""
        if isinstance(node, (list, tuple)):
            if len(node) == 0:
                return None
            node0 = node[0]
            node1 = node[-1]
        else:
            node0 = node
            node1 = node
        if hasattr(node0, 'lineno') and hasattr(node0, 'col_offset'):
            li1, col1 = node0.first_token.start
            li2, col2 = node1.last_token.end
            li1 += self.line_offset - 1
            li2 += self.line_offset - 1
            col1 += self.col_offset
            col2 += self.col_offset
            return Location(self.filename,
                            li1, col1, li2, col2, node)
        else:
            # Some nodes like Index carry no location information, but
            # we basically just pass through them.
            return None  # pragma: no cover

    def make_condition_blocks(self, block):
        """Make two blocks for an if statement or expression."""
        with About(block.graph.debug, 'if_true'):
            true_block = Block(self, auxiliary=True, flatten_inference=True)
        true_block.preds.append(block)
        true_block.mature()

        with About(block.graph.debug, 'if_false'):
            false_block = Block(self, auxiliary=True, flatten_inference=True)
        false_block.preds.append(block)
        false_block.mature()

        return true_block, false_block

    def parse(self) -> Graph:
        """Parse the function into a Myia graph."""
        src0 = inspect.getsource(self.function)
        src = textwrap.dedent(src0)
        # We need col_offset to compensate for the dedent
        self.col_offset = len(src0.split('\n')[0]) - len(src.split('\n')[0])
        tree = asttokens.ASTTokens(src, parse=True).tree
        function_def = tree.body[0]
        assert isinstance(function_def, ast.FunctionDef)
        graph = self._process_function(None, function_def).graph
        return graph

    def process_FunctionDef(self, block: 'Block',
                            node: ast.FunctionDef) -> 'Block':
        """Process a function definition.

        Args:
            block: Predecessor block (optional). If given, this is a nested
                function definition.
            node: The function definition.

        """
        function_block = self._process_function(block, node)
        block.write(node.name, Constant(function_block.graph))
        return block

    def _process_function(self, block: Optional['Block'],
                          node: ast.FunctionDef) -> Tuple['Block', 'Block']:
        """Process a function definition and return first and final blocks."""
        function_block = Block(self)
        if block:
            function_block.preds.append(block)
        else:
            # This is the top-level function, so we set self.graph
            self.graph = function_block.graph

        function_block.mature()
        function_block.graph.debug.name = node.name
        if node.args.kwarg is not None or node.args.kwonlyargs != []:
            raise NotImplementedError("No support for keyword arguments")
        if node.args.vararg:
            raise NotImplementedError("No support for varargs")
        for arg in node.args.args:
            with DebugInherit(ast=arg, location=self.make_location(arg)):
                anf_node = Parameter(function_block.graph)
            anf_node.debug.name = arg.arg
            function_block.graph.parameters.append(anf_node)
            function_block.write(arg.arg, anf_node)
        function_block.write(node.name, Constant(function_block.graph))
        self.process_statements(function_block, node.body)
        if function_block.graph.return_ is None:
            raise MyiaSyntaxError("Function doesn't return a value",
                                  self.make_location(node))
        # TODO: check that if after_block returns?
        return function_block

    @pyoverload
    def process_node(self, block: 'Block', node: ast.expr) -> ANFNode:
        pass

    @pyoverload  # noqa
    def process_node(self, block: 'Block', node: ast.slice) -> ANFNode:
        pass

    @pyoverload  # noqa
    def process_node(self, block: 'Block', node: ast.stmt) -> 'Block':
        pass

    def process_node(self, block, node):  # noqa
        """Process an ast node."""
        method_name = f'process_{node.__class__.__name__}'
        method = getattr(self, method_name, None)
        if method:
            with DebugInherit(ast=node, location=self.make_location(node)):
                return method(block, node)
        else:
            raise NotImplementedError(node)  # pragma: no cover

    # Expression implementations

    def process_BinOp(self, block: 'Block', node: ast.BinOp) -> ANFNode:
        """Process binary operators: `a + b`, `a | b`, etc."""
        func = block._resolve_ast_type(node.op)
        left = self.process_node(block, node.left)
        right = self.process_node(block, node.right)
        return Apply([func, left, right], block.graph)

    def process_UnaryOp(self, block: 'Block', node: ast.UnaryOp) -> ANFNode:
        """Process unary operators: `+a`, `-a`, etc."""
        func = block._resolve_ast_type(node.op)
        operand = self.process_node(block, node.operand)
        return Apply([func, operand], block.graph)

    def process_Compare(self, block: 'Block', node: ast.Compare) -> ANFNode:
        """Process comparison operators: `a == b`, `a > b`, etc."""
        ops = [block._resolve_ast_type(op) for op in node.ops]
        assert len(ops) == 1
        left = self.process_node(block, node.left)
        right = self.process_node(block, node.comparators[0])
        return Apply([ops[0], left, right], block.graph)

    def process_BoolOp(self, block: 'Block', node: ast.BinOp) -> ANFNode:
        """Process boolean operators: `a and b`, `a or b`."""
        def fold(block, values, mode):
            first, *rest = values
            test = self.process_node(block, first)
            if rest:
                true_block, false_block = self.make_condition_blocks(block)

                if mode == 'and':
                    b1, b2 = true_block, false_block
                else:
                    b1, b2 = false_block, true_block

                b1.graph.output = fold(b1, rest, mode)
                b2.graph.output = test

                switch = block.graph.apply(block.operation('switch'),
                                           block.force_bool(test),
                                           true_block.graph,
                                           false_block.graph)
                return block.graph.apply(switch)
            else:
                return test

        init, *rest = node.values
        if isinstance(node.op, ast.And):
            return fold(block, node.values, 'and')
        elif isinstance(node.op, ast.Or):
            return fold(block, node.values, 'or')
        else:
            raise AssertionError(f'Unknown BoolOp: {node.op}')

    def process_IfExp(self, block: 'Block', node: ast.IfExp) -> ANFNode:
        """Process if expression: `a if b else c`."""
        cond = self.process_node(block, node.test)

        true_block, false_block = self.make_condition_blocks(block)
        true_block.graph.debug.location = self.make_location(node.body)
        false_block.graph.debug.location = self.make_location(node.orelse)

        tb = self.process_node(true_block, node.body)
        fb = self.process_node(false_block, node.orelse)

        tg = true_block.graph
        fg = false_block.graph

        tg.output = tb
        fg.output = fb

        switch = block.graph.apply(block.operation('switch'), cond, tg, fg)
        return block.graph.apply(switch)

    def process_Lambda(self, block: 'Block', node: ast.Lambda) -> ANFNode:
        """Process lambda: `lambda x, y: x + y`."""
        function_block = Block(self)
        function_block.preds.append(block)
        function_block.mature()

        for arg in node.args.args:
            with DebugInherit(ast=arg, location=self.make_location(arg)):
                anf_node = Parameter(function_block.graph)
            anf_node.debug.name = arg.arg
            function_block.graph.parameters.append(anf_node)
            function_block.write(arg.arg, anf_node)

        function_block.graph.output = \
            self.process_node(function_block, node.body)
        return Constant(function_block.graph)

    def process_Name(self, block: 'Block', node: ast.Name) -> ANFNode:
        """Process variables: `variable_name`."""
        return block.read(node.id)

    def process_Num(self, block: 'Block', node: ast.Num) -> ANFNode:
        """Process numbers: `1`, `2.5`, etc."""
        return Constant(node.n)

    def process_Str(self, block: 'Block', node: ast.Str) -> ANFNode:
        """Process strings: `"a"`, `'hello world'`, etc."""
        return Constant(node.s)

    def process_Call(self, block: 'Block', node: ast.Call) -> ANFNode:
        """Process function calls: `f(x)`, etc."""
        func = self.process_node(block, node.func)
        args = [self.process_node(block, arg) for arg in node.args]
        return Apply([func] + args, block.graph)

    def process_NameConstant(self, block: 'Block',
                             node: ast.NameConstant) -> ANFNode:
        """Process special constants: `True`, `False`, `None`, etc."""
        return Constant(node.value)

    def process_Tuple(self, block: 'Block', node: ast.Tuple) -> ANFNode:
        """Process tuple literals."""
        op = block.operation('make_tuple')
        elts = [self.process_node(block, e) for e in node.elts]
        if len(elts) == 0:
            return Constant(())
        else:
            return block.graph.apply(op, *elts)

    def process_List(self, block: 'Block', node: ast.List) -> ANFNode:
        """Process list literals."""
        op = block.operation('make_list')
        elts = [self.process_node(block, e) for e in node.elts]
        return block.graph.apply(op, *elts)

    def process_Subscript(self, block: 'Block',
                          node: ast.Subscript) -> ANFNode:
        """Process subscripts: `x[y]`."""
        op = block.operation('getitem')
        value = self.process_node(block, node.value)
        slice = self.process_node(block, node.slice)
        return Apply([op, value, slice], block.graph)

    def process_Index(self, block: 'Block', node: ast.Index) -> ANFNode:
        """Process subscript indexes."""
        return self.process_node(block, node.value)

    def process_Attribute(self, block: 'Block',
                          node: ast.Attribute) -> ANFNode:
        """Process attributes: `x.y`."""
        op = block.operation('getattr')
        value = self.process_node(block, node.value)
        return Apply([op, value, Constant(node.attr)], block.graph)

    # Statement implementations

    def process_statements(self, block: 'Block',
                           nodes: List[ast.stmt]) -> 'Block':
        """Process a sequence of statements.

        If the list of statements is empty, a new empty block will be
        constructed and returned. This ensures that empty code blocks (e.g. an
        empty else branch) still have a corresponding block that can be used to
        call the continuation from.

        """
        for node in nodes:
            block = self.process_node(block, node)
        return block

    def process_Return(self, block: 'Block', node: ast.Return) -> 'Block':
        """Process a return statement."""
        inputs = [Constant(primops.return_),
                  self.process_node(block, node.value)]
        return_ = Apply(inputs, block.graph)
        assert block.graph.return_ is None
        block.graph.return_ = return_
        return block

    def process_Assign(self, block: 'Block', node: ast.Assign) -> 'Block':
        """Process an assignment."""
        anf_node = self.process_node(block, node.value)

        def write(targ, anf_node):

            if isinstance(targ, ast.Name):
                # CASE: x = value
                anf_node.debug.name = targ.id
                if anf_node.is_constant_graph():
                    if anf_node.value.debug.name is None:
                        anf_node.value.debug.name = targ.id
                block.write(targ.id, anf_node)

            elif isinstance(targ, ast.Tuple):
                # CASE: x, y = value
                for i, elt in enumerate(targ.elts):
                    op = block.operation('getitem')
                    new_node = Apply([op, anf_node, Constant(i)], block.graph)
                    write(elt, new_node)

            else:
                raise NotImplementedError(node.targets)  # pragma: no cover

        for targ in node.targets:
            write(targ, anf_node)

        return block

    def process_Expr(self, block: 'Block', node: ast.Expr) -> 'Block':
        """Process an expression statement.

        This ignores the statement.
        """
        return block

    def process_If(self, block: 'Block', node: ast.If) -> 'Block':
        """Process a conditional statement.

        A conditional statement generates 3 functions: The true branch, the
        false branch, and the continuation.
        """
        # Process the condition
        cond = self.process_node(block, node.test)
        cond = block.force_bool(cond)

        # Create two branches
        true_block, false_block = self.make_condition_blocks(block)
        true_block.graph.debug.location = self.make_location(node.body)
        false_block.graph.debug.location = self.make_location(node.orelse)

        # Create the continuation
        with About(block.graph.debug, 'if_after'):
            after_block = Block(self, auxiliary=True, flatten_inference=True)

        # Process the first branch
        true_end = self.process_statements(true_block, node.body)
        # A return statement in the branch might mean that a continuation has
        # already been set
        if not true_end.graph.return_:
            true_end.jump(after_block)

        # And the second
        false_end = self.process_statements(false_block, node.orelse)
        if not false_end.graph.return_:
            false_end.jump(after_block)

        # And stich it together
        block.cond(cond, true_block, false_block)
        after_block.mature()
        return after_block

    def process_While(self, block: 'Block', node: ast.While) -> 'Block':
        """Process a while loop.

        A while loop will generate 3 functions: The test, the body, and the
        continuation.

        """
        with About(block.graph.debug, 'while_header'):
            header_block = Block(self, auxiliary=True)
        with About(block.graph.debug, 'while_body'):
            body_block = Block(self, auxiliary=True)
        with About(block.graph.debug, 'while_after'):
            after_block = Block(self, auxiliary=True)
        body_block.preds.append(header_block)
        after_block.preds.append(header_block)
        block.jump(header_block)
        cond = self.process_node(header_block, node.test)
        body_block.mature()
        header_block.cond(cond, body_block, after_block)
        after_body = self.process_statements(body_block, node.body)
        if not after_body.graph.return_:
            after_body.jump(header_block)
        header_block.mature()
        after_block.mature()
        return after_block

    def process_For(self, block: 'Block', node: ast.For) -> 'Block':
        """Process a for loop.

        ```
        for x in xs:
            body
        ```

        Is essentially compiled as:

        ```
        it = iter(xs)
        while hasnext(it):
            x, it = next(it)
            body
        ```

        A for loop will generate 3 functions: The test, the body, and the
        continuation.
        """
        op_iter = block.operation('iter')
        op_next = block.operation('next')
        op_getitem = block.operation('getitem')
        op_hasnext = block.operation('hasnext')

        # Initialization of the iterator, only done once
        init = block.graph.apply(op_iter,
                                 self.process_node(block, node.iter))

        # Checks hasnext on the iterator.
        with About(block.graph.debug, 'for_header'):
            header_block = Block(self, auxiliary=True)
        # We explicitly add the iterator as the first argument
        it = header_block.graph.add_parameter()
        cond = header_block.graph.apply(op_hasnext, it)

        # Body of the iterator.
        with About(block.graph.debug, 'for_body'):
            body_block = Block(self, auxiliary=True)
        body_block.preds.append(header_block)
        # app = next(it); target = app[0]; it = app[1]
        app = body_block.graph.apply(op_next, it)
        target = body_block.graph.apply(op_getitem, app, 0)
        target.debug.name = node.target.id
        it2 = body_block.graph.apply(op_getitem, app, 1)
        # We link the variable name to the target
        body_block.write(node.target.id, target)
        # We set some debug data on all iterator variables
        it_debug_data = About(target.debug, 'iterator')
        it.debug.about = it_debug_data
        it2.debug.about = it_debug_data
        init.debug.about = it_debug_data

        # This is the block after for
        with About(block.graph.debug, 'for_after'):
            after_block = Block(self, auxiliary=True)
        after_block.preds.append(header_block)

        block.jump(header_block, init)

        body_block.mature()
        header_block.cond(cond, body_block, after_block)

        after_body_block = self.process_statements(body_block, node.body)
        if not after_body_block.graph.return_:
            after_body_block.jump(header_block, it2)

        header_block.mature()
        after_block.mature()
        return after_block


class Block:
    """A basic block.

    A basic block is used while parsing the Python code to represent a segment
    of code (e.g. a function body, loop body, a branch). During parsing it
    keeps track of a variety of information needed to resolve variable names.

    Attributes:
        variables: A mapping from variable names to the nodes representing the
            bound value at this point of parsing. If a variable name is not in
            this mapping, it needs to be resolved in the predecessor blocks.
        phi_nodes: A mapping from parameter nodes corresponding to phi nodes to
            variable names. Once all the predecessor blocks (calling functions)
            are known, we can resolve these variable names in the predecessor
            blocks to find out what the arguments at the call site are.
        jumps: A mapping from successor blocks to the function calls that
            correspond to these jumps. This is information that was not used in
            the FIRM algorithm; it is necessary here because it is not possible
            to distinguish regular function calls from the tail calls used for
            control flow.
        matured: Whether all the predecessors of this block have been
            constructed yet. If a block is not mature and a variable cannot be
            resolved, we have to construct a phi node (i.e. add a parameter to
            this function). Once the block is mature, we will resolve the
            variable in the parent blocks and use them as arguments.
        preds: The predecessor blocks of this block.
        graph: The ANF function graph corresponding to this basic block.

    """

    def __init__(self, parser: Parser, **flags) -> None:
        """Construct a basic block.

        Constructing a basic block also constructs a corresponding function,
        and a constant that can be used to call this function.

        Arguments:
            parser: The Parser object.
            flags: Flags to give to that Block's graph.
        """
        self.parser = parser

        self.matured: bool = False
        self.variables: Dict[str, ANFNode] = {}
        self.preds: List[Block] = []
        self.phi_nodes: Dict[Parameter, str] = {}
        self.jumps: Dict[Block, Apply] = {}
        self.graph: Graph = Graph()
        self.graph.flags.update(flags)

    def set_phi_arguments(self, phi: Parameter) -> None:
        """Resolve the arguments to a phi node.

        Args:
            phi: The `Parameter` node which is functioning as a phi node. The
                arguments corresponding to this parameter will be read from
                predecessor blocks (functions).

        """
        varnum = self.phi_nodes[phi]
        for pred in self.preds:
            arg = pred.read(varnum)
            jump = pred.jumps[self]
            jump.inputs.append(arg)
        # TODO remove_unnecessary_phi(phi)

    def mature(self) -> None:
        """Mature this block.

        A block is matured once all of its predecessor blocks have been
        processed. This triggers the resolution of phi nodes.

        """
        # Use the function parameters to ensure proper ordering.
        for phi in self.graph.parameters:
            if phi in self.phi_nodes:
                self.set_phi_arguments(phi)
        self.matured = True

    def _resolve_ast_type(self, op):
        return self.make_resolve(*ast_map[type(op)])

    def make_resolve(self, module_name, symbol_name):
        """Return a subtree that resolves a name in a module."""
        return self.graph.apply(
            primops.resolve,
            Constant(module_name),
            Constant(symbol_name)
        )

    def operation(self, symbol_name):
        """Return a subtree that resolves a name in the operations module."""
        return self.make_resolve(operations_ns, symbol_name)

    def force_bool(self, cond):
        """Wrap a condition in a call to bool()."""
        rval = self.graph.apply(
            self.operation('bool'),
            cond
        )
        rval.debug.location = cond.debug.location
        return rval

    def read(self, varnum: str) -> ANFNode:
        """Read a variable.

        If this name has defined given in one of the previous statements, it
        will be trivially resolved. It is possible that the variable was
        defined in a previous block (e.g. outside of the loop body or the
        branch). In this case, it will be resolved only if all predecessor
        blocks are available. If they are not, we will assume that this
        variable is given as a function argument (which plays the role of a phi
        node).

        Args:
            varnum: The name of the variable to read.

        """
        if varnum in self.variables:
            return _fresh(self.variables[varnum])
        if self.matured:
            if len(self.preds) == 1:
                return self.preds[0].read(varnum)
            elif not self.preds:
                cn = self.parser.closure_namespace
                if varnum in cn:
                    return self.make_resolve(cn, varnum)
                else:
                    ns = self.parser.global_namespace
                    return self.make_resolve(ns, varnum)
        # TODO: point to the original definition
        with About(NamedDebugInfo(name=varnum), 'phi'):
            phi = Parameter(self.graph)
        self.graph.parameters.append(phi)
        self.phi_nodes[phi] = varnum
        self.write(varnum, phi)
        if self.matured:
            self.set_phi_arguments(phi)
        return phi

    def write(self, varnum: str, node: ANFNode) -> None:
        """Write a variable.

        When assignment is used to bound a value to a name, we store this
        mapping in the block to be used by subsequent statements.

        Args:
            varnum: The name of the variable to store.
            node: The node representing this value.

        """
        self.variables[varnum] = node

    def jump(self, target: 'Block', *args) -> Apply:
        """Jumping from one block to the next becomes a tail call.

        This method will generate the tail call by calling the graph
        corresponding to the target block using an `Apply` node, and returning
        its value with a `Return` node. It will update the predecessor blocks
        of the target appropriately.

        Args:
            target: The block to jump to from this statement.

        """
        assert self.graph.return_ is None
        jump = self.graph.apply(target.graph, *args)
        self.jumps[target] = jump
        target.preds.append(self)
        self.graph.output = jump

    def cond(self, cond: ANFNode, true: 'Block', false: 'Block') -> Apply:
        """Perform a conditional jump.

        This method will generate the call to the if expression and return its
        value. The predecessor blocks of the branches will be updated
        appropriately.

        Args:
            cond: The node representing the condition (true or false).
            true: The block to jump to if the condition is true.
            false: The block to jump to if the condition is false.

        """
        assert self.graph.return_ is None
        switch = self.graph.apply(
            self.operation('switch'),
            cond,
            true.graph,
            false.graph
        )
        self.graph.output = self.graph.apply(switch)
