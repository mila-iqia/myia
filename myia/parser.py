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

The parsing of Python functions is handled on 3 separate levels, mimicking the
way that Python resolves variable names (local/enclosed scope, globals,
builtins):

Global environment
  There is usually one environment per Python process, although multiple can be
  instantiated. Within a single environment, the same Python objects will
  correspond to the same Myia object e.g. if two functions use the same
  subroutine and are parsed, this subroutine will only be parsed once.

Parser
  There is one parser per user-defined function. A parser is responsible for
  e.g. resolving variable names using the function's global scope and managing
  the parsing process.

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
import inspect
import textwrap
import operator
from types import FunctionType
from typing import \
    overload, Any, Dict, List, Optional, Tuple, Type, NamedTuple

from myia.anf_ir import ANFNode, Parameter, Apply, Graph, Constant
from .info import DebugInherit, About


class Location(NamedTuple):
    """A location in source code.

    Attributes:
        filename: The filename.
        line: The line number.
        column: The column number.

    """

    filename: str
    line: str
    column: int


class Environment:
    """Environment to parse a function in.

    An environment consists of a mapping from Python objects to nodes. If a
    Python object is missing from this mapping, the environment will try to
    convert it into a Myia object e.g. create a Myia constant or parse a Python
    function into a Myia function.

    The environment is also in charge of resolving variable names that could
    not be resolved in the function's local and global namespace i.e. for
    built-ins and for undefined variable names.

    Lastly, the environment has a mapping from AST nodes to Myia nodes that the
    parser uses e.g. for mapping binary operators to the correct primitives.

    Functions parsed in different environments will have no nodes in common.
    Functions parsed in the same environment will use the same Myia objects for
    the same Python objects.

    """

    def __init__(self, object_map: Dict[int, ANFNode],
                 ast_map: Dict[Type[ast.AST], ANFNode]) -> None:
        """Construct an environment.

        Args:
            object_map: The object map maps Python object id's to their
                corresponding Myia nodes. The dictionary uses ids instead of
                the object itself so that different objects that are equal
                don't have to share a node.
            ast_map: A mapping from Python AST nodes to corresponding Myia
                nodes.

        """
        self.object_map = object_map
        self.ast_map = ast_map

    def map(self, obj: Any) -> ANFNode:
        """Map a Python object to an ANF node.

        If the object was already converted, the existing Myia node will be
        returned.

        """
        if id(obj) in self.object_map:
            return self.object_map[id(obj)]
        if isinstance(obj, (bool, float, int, str)):
            node = Constant(obj)
            self.object_map[id(obj)] = node
            return node
        if isinstance(obj, FunctionType):
            parser = Parser(self, obj)
            graph = parser.parse()
            node = Constant(graph)
            return node
        raise ValueError(obj)


class Parser:
    """Parser for a function.

    This class handles the parsing of a single user-defined function. It is
    responsible for handling e.g. globals.

    Todo:
        This parser resolves globals in the current Python namespace. This
        causes two problems: Forward declarations are not support, and changes
        to Python's mutable namespace are not reflected if the function is not
        reparsed.

        In the future we will have an implementation that resolves globals
        just-in-time (to support forward declarations) and acts accordingly
        when the Python namespace changes.

    """

    def __init__(self, environment: Environment,
                 function: FunctionType,
                 globals_: Dict[str, Any] = None) -> None:
        """Construct a parser."""
        self.environment = environment
        self.function = function
        _, self.line_offset = inspect.getsourcelines(function)
        self.filename: str = inspect.getfile(function)
        self.block_map: Dict[Block, Constant] = {}
        if globals_ is None:
            free_vars = inspect.getclosurevars(function)  # type: ignore
            assert isinstance(free_vars.builtins, Dict)
            globals_ = free_vars.builtins
            globals_.update(free_vars.globals)
            globals_.update(free_vars.nonlocals)
        self.globals_ = globals_

    def make_location(self, node: ast.AST) -> Location:
        """Create a Location from an AST node."""
        if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
            return Location(self.filename,
                            node.lineno + self.line_offset - 1,  # type: ignore
                            node.col_offset)  # type: ignore
        else:
            # Some nodes like Index carry no location information, but
            # we basically just pass through them.
            return None  # pragma: no cover

    def parse(self) -> Graph:
        """Parse the function into a Myia graph."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(self.function)))
        function_def = tree.body[0]
        assert isinstance(function_def, ast.FunctionDef)
        return self._process_function(None, function_def)[1].graph

    def get_block_function(self, block: 'Block') -> Constant:
        """Return node representing the function corresponding to a block."""
        if block in self.block_map:
            return self.block_map[block]
        node = Constant(block.graph)
        self.block_map[block] = node
        return node

    def read(self, varnum: str) -> ANFNode:
        """Read a variable from the function's global namespace."""
        assert isinstance(self.function, FunctionType)
        if varnum in self.globals_:
            return self.environment.map(self.globals_[varnum])
        raise ValueError(varnum)

    def process_FunctionDef(self, block: 'Block',
                            node: ast.FunctionDef) -> 'Block':
        """Process a function definition.

        Args:
            block: Predecessor block (optional). If given, this is a nested
                function definition.
            node: The function definition.

        """
        _, function_block = self._process_function(block, node)
        block.write(node.name, self.get_block_function(function_block))
        return block

    def _process_function(self, block: Optional['Block'],
                          node: ast.FunctionDef) -> Tuple['Block', 'Block']:
        """Process a function definition and return first and final blocks."""
        function_block = Block(self)
        # Add this mapping immediately so that recursive calls can be resolved
        self.environment.object_map[id(self.function)] = \
            self.get_block_function(function_block)
        if block:
            function_block.preds.append(block)
        function_block.mature()
        function_block.graph.debug.name = node.name
        for arg in node.args.args:
            with DebugInherit(ast=arg, location=self.make_location(arg)):
                anf_node = Parameter(function_block.graph)
            anf_node.debug.name = arg.arg
            function_block.graph.parameters.append(anf_node)
            function_block.write(arg.arg, anf_node)
        function_block.write(node.name,
                             self.get_block_function(function_block))
        final_block = self.process_statements(function_block, node.body)
        return final_block, function_block

    @overload
    def process_node(self, block: 'Block', node: ast.expr) -> ANFNode:
        pass

    @overload  # noqa
    def process_node(self, block: 'Block', node: ast.slice) -> ANFNode:
        pass

    @overload  # noqa
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
        func = self.environment.ast_map[type(node.op)]
        left = self.process_node(block, node.left)
        right = self.process_node(block, node.right)
        return Apply([func, left, right], block.graph)

    def process_UnaryOp(self, block: 'Block', node: ast.UnaryOp) -> ANFNode:
        """Process unary operators: `+a`, `-a`, etc."""
        func = self.environment.ast_map[type(node.op)]
        operand = self.process_node(block, node.operand)
        return Apply([func, operand], block.graph)

    def process_Compare(self, block: 'Block', node: ast.Compare) -> ANFNode:
        """Process comparison operators: `a == b`, `a > b`, etc."""
        ops = [self.environment.ast_map[type(op)] for op in node.ops]
        assert len(ops) == 1
        left = self.process_node(block, node.left)
        right = self.process_node(block, node.comparators[0])
        return Apply([ops[0], left, right], block.graph)

    def process_Name(self, block: 'Block', node: ast.Name) -> ANFNode:
        """Process variables: `variable_name`."""
        return block.read(node.id)

    def process_Num(self, block: 'Block', node: ast.Num) -> ANFNode:
        """Process numbers: `1`, `2.5`, etc."""
        return self.environment.map(node.n)

    def process_Call(self, block: 'Block', node: ast.Call) -> ANFNode:
        """Process function calls: `f(x)`, etc."""
        func = self.process_node(block, node.func)
        args = [self.process_node(block, arg) for arg in node.args]
        return Apply([func] + args, block.graph)

    def process_NameConstant(self, block: 'Block',
                             node: ast.NameConstant) -> ANFNode:
        """Process special constants: `True`, `False`, `None`, etc."""
        return self.environment.map(node.value)

    def process_Tuple(self, block: 'Block', node: ast.Tuple) -> ANFNode:
        """Process tuple literals."""
        op = self.environment.ast_map[ast.Tuple]
        elts = [self.process_node(block, e) for e in node.elts]
        return Apply([op, *elts], block.graph)

    def process_Subscript(self, block: 'Block',
                          node: ast.Subscript) -> ANFNode:
        """Process subscripts: `x[y]`."""
        op = self.environment.ast_map[ast.Subscript]
        value = self.process_node(block, node.value)
        slice = self.process_node(block, node.slice)
        return Apply([op, value, slice], block.graph)

    def process_Index(self, block: 'Block', node: ast.Index) -> ANFNode:
        """Process subscripts with simple index: `x[y]`."""
        return self.process_node(block, node.value)

    def process_Attribute(self, block: 'Block',
                          node: ast.Attribute) -> ANFNode:
        """Process attributes: `x.y`."""
        op = self.environment.ast_map[ast.Attribute]
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
        inputs = [self.environment.ast_map[ast.Return],
                  self.process_node(block, node.value)]
        return_ = Apply(inputs, block.graph)
        block.graph.return_ = return_
        return block

    def process_Assign(self, block: 'Block', node: ast.Assign) -> 'Block':
        """Process an assignment."""
        anf_node = self.process_node(block, node.value)
        targ, = node.targets

        def write(targ, anf_node):

            if isinstance(targ, ast.Name):
                # CASE: x = value
                anf_node.debug.name = targ.id
                block.write(targ.id, anf_node)

            elif isinstance(targ, ast.Tuple):
                # CASE: x, y = value
                for i, elt in enumerate(targ.elts):
                    op = self.environment.ast_map[ast.Subscript]
                    new_node = Apply([op, anf_node, Constant(i)], block.graph)
                    write(elt, new_node)

            elif isinstance(targ, ast.Subscript):
                if isinstance(targ.value, ast.Name):
                    # CASE: x[y] = value
                    op = self.environment.object_map[id(operator.setitem)]
                    obj = self.process_node(block, targ.value)
                    idx = self.process_node(block, targ.slice)
                    new_node = Apply([op, obj, idx, anf_node], block.graph)
                    write(targ.value, new_node)
                else:
                    # UNSUPPORTED: f()[x] = value
                    raise NotImplementedError(
                        "You can only set a slice on a variable."
                    )  # pragma: no cover

            elif isinstance(targ, ast.Attribute):
                if isinstance(targ.value, ast.Name):
                    # CASE: x.y = value
                    op = self.environment.object_map[id(setattr)]
                    obj = self.process_node(block, targ.value)
                    idx = Constant(targ.attr)
                    new_node = Apply([op, obj, idx, anf_node], block.graph)
                    write(targ.value, new_node)
                else:
                    # UNSUPPORTED: f().x = value
                    raise NotImplementedError(
                        "You can only set an attribute on a variable."
                    )  # pragma: no cover

            else:
                raise NotImplementedError(node.targets)  # pragma: no cover

        write(targ, anf_node)
        return block

    def process_AugAssign(self, block: 'Block',
                          node: ast.AugAssign) -> 'Block':
        """Process an augmented assignment `x += y`."""
        # We just transform it into an Assign node that applies a BinOp,
        # i.e. the AST for `x = x + y`
        # This may repeat computations, e.g. for `x[a * a] += 1` it will
        # create two multiplication nodes, but if we integrate CSE later
        # that problem will go away on its own.
        app = ast.BinOp()
        app.op = node.op
        app.left = node.target
        app.right = node.value
        ass = ast.Assign()
        ass.targets = [node.target]
        ass.value = app
        return self.process_Assign(block, ass)

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

        # Create two branches
        with About(block.graph.debug, 'if_true'):
            true_block = Block(self)
        with About(block.graph.debug, 'if_false'):
            false_block = Block(self)
        true_block.preds.append(block)
        false_block.preds.append(block)
        true_block.mature()
        false_block.mature()

        # Create the continuation
        with About(block.graph.debug, 'if_after'):
            after_block = Block(self)

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
            header_block = Block(self)
        with About(block.graph.debug, 'while_body'):
            body_block = Block(self)
        with About(block.graph.debug, 'while_after'):
            after_block = Block(self)
        body_block.preds.append(header_block)
        after_block.preds.append(header_block)
        block.jump(header_block)
        cond = self.process_node(header_block, node.test)
        body_block.mature()
        header_block.cond(cond, body_block, after_block)
        after_body = self.process_statements(body_block, node.body)
        after_body.jump(header_block)
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

    def __init__(self, parser: Parser) -> None:
        """Construct a basic block.

        Constructing a basic block also constructs a corresponding function,
        and a constant that can be used to call this function.

        """
        self.parser = parser

        self.matured: bool = False
        self.variables: Dict[str, ANFNode] = {}
        self.preds: List[Block] = []
        self.phi_nodes: Dict[Parameter, str] = {}
        self.jumps: Dict[Block, Apply] = {}
        self.graph: Graph = Graph()

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
            return self.variables[varnum]
        if self.matured:
            if len(self.preds) == 1:
                return self.preds[0].read(varnum)
            elif not self.preds:
                return self.parser.read(varnum)
        # TODO: point to the original definition
        with About(DebugInherit(name=varnum), 'phi'):
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

    def jump(self, target: 'Block') -> Apply:
        """Jumping from one block to the next becomes a tail call.

        This method will generate the tail call by calling the graph
        corresponding to the target block using an `Apply` node, and returning
        its value with a `Return` node. It will update the predecessor blocks
        of the target appropriately.

        Args:
            target: The block to jump to from this statement.

        """
        jump = Apply([self.parser.get_block_function(target)], self.graph)
        self.jumps[target] = jump
        target.preds.append(self)
        inputs = [self.parser.environment.ast_map[ast.Return], jump]
        return_ = Apply(inputs, self.graph)
        self.graph.return_ = return_
        return return_

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
        inputs = [self.parser.environment.ast_map[ast.If], cond,
                  self.parser.get_block_function(true),
                  self.parser.get_block_function(false)]
        if_ = Apply(inputs, self.graph)
        inputs = [self.parser.environment.ast_map[ast.Return], if_]
        return_ = Apply(inputs, self.graph)
        self.graph.return_ = return_
        return return_
