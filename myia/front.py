"""
Parse Python's AST to produce a Myia AST.

* Parser does the transform.
* parse_source, parse_function are the nicer API entry points.

== Understanding the output

The Parser rewrites Python to a pure functional form, where loops
are mapped to recursive calls. Evaluation is eager, therefore
the branches of conditionals are also mapped to functions.

=== Conditionals

An if statement is, more or less, transformed as follows:

    if test():
        return f(x)
    else:
        return g(y)

Becomes:

    [lambda: g(y), lambda: f(x)][test()]()

Recall that True and False are equivalent to 1 and 0 respectively.
In any case, Myia uses an operation called ``switch``, which is like
a non-lazy if, but the principle is the same.

=== Loops

A loop is transformed like this:

    while test():
        x = f(y)
        y = g(z)

Becomes:

    def while_fn(x, y):
        if test():
            x = f(y)
            y = g(z)
            return while_fn(x, y)
        else:
            return x, y
    x, y = while_fn(x, y)

The ``if`` statement is also transformed, as explained earlier, so
a while loop actually becomes two functions.

=== Variables

The result is a form of SSA, that is to say, all variables are
set exactly once. Multiple assignments to the same variable in Python
become assignments to new variables:

    x = 1; x = 2 * x; x = x + x
    ==>
    x = 1; x#2 = 2 * x; x#3 = x#2 + x#2

=== Free variables

Myia's transform produces closure-converted functions. What this means
is that functions have no free variables. Instead, we add the free
variables as arguments, and we create a data structure (a closure
object, which is like a partial application) that contains a pointer
to the function and to a list of values that we can prepend to its
arguments. In other words:

    def f(x):
        def g(y):
            return x + y
        return g

Becomes something like this:

    def _g(x, y):
        return x + y

    def f(x):
        g = functools.partial(_g, x)
        return g

So the idea is that we pull _g out and we make it a global function
(with a name chosen to avoid clashes), but then we create a partial
application object where we can store x. This is equivalent to our
original g.

=== Auxiliary function names

In order to enhance readability somewhat, the auxiliary functions that
the parser creates follow this naming scheme:

* ✓f is executed when the condition of the first ``if`` statement of
  f is True.
* ✗f is executed when the condition of the first ``if`` statement of
  f is False.
* ⤾f is the test for a ``while`` loop. Remember it by the fact it's
  only half a loop. We're only trying to figure out whether to
  perform a loop iteration or not.
* ⥁f is the body for a ``while`` loop. Remember it by the fact it's
  a complete loop, so at the end of it we performed one full
  iteration.
* ✓f#2 is the condition of the second ``if`` statement, and so on.

Other transforms may create other derived functions and variables.
See ``grad.py`` for gradient-related annotations.
"""


from typing import \
    Dict, Set, List, Tuple as TupleT, \
    cast, Union, Callable, Optional, Any

from .buche import buche
from .event import EventDispatcher
from collections import OrderedDict
from .ast import \
    MyiaASTNode, \
    Location, Symbol, Value, \
    Let, Lambda, Apply, \
    Begin, Tuple, Closure, _Assign, \
    GenSym, ParseEnv, About, VariableTracker
from .util import group_contiguous
from .symbols import \
    get_operator, builtins, HIDGLOB, \
    THEN, ELSE, WTEST, WLOOP, LBDA
from uuid import uuid4 as uuid
import ast
from copy import copy
import inspect
import textwrap
import sys


class MyiaSyntaxError(Exception):
    """
    Class for syntax errors in Myia. This exception type should be
    raised for any feature that is not supported.

    Attributes:
        location: The error's location in the original source.
        message: A precise assessment of the problem.
    """
    def __init__(self, location: Location, message: str) -> None:
        self.location = location
        self.message = message


_prevhook = sys.excepthook


def _exception_handler(exception_type, exception, traceback):
    # We override the default exception handler so that it prints
    # MyiaSyntaxErrors like a SyntaxError. Not sure how else to do
    # it.
    # TODO: actually, maybe inherit from SyntaxError? Investigate.
    if exception_type == MyiaSyntaxError:
        print(
            "{}: {}".format(exception_type.__name__, exception.message),
            file=sys.stderr
        )
        if exception.location:
            print(exception.location.traceback(), file=sys.stderr)
    else:
        _prevhook(exception_type, exception, traceback)


sys.excepthook = _exception_handler


class Locator:
    """
    Call a Locator instance on a Python AST node to get an
    ``ast.Location`` instance.

    When parsing a function string, locations in the
    Python AST will start at line 1. Feed ``line_offset`` to
    the constructor to compensate if you want to get correct line
    numbers.

    Attributes:
        url: The source code's filename.
        line_offset: The line offset at which the code starts.
    """
    def __init__(self, url: str, line_offset: int) -> None:
        self.url = url
        self.line_offset = line_offset

    def __call__(self, node: Union[ast.expr, ast.stmt]) -> Location:
        try:
            return Location(
                self.url,
                node.lineno + self.line_offset - 1,
                node.col_offset,
                node
            )
        except AttributeError:
            return None


class LocVisitor:
    """
    Base class to visit Python's AST and map it to MyiaASTNodes.

    ``LocVisitor::visit`` automatically transfers source code
    line/column information from one to the other.

    Subclasses should override ``visit_<class_name>``, e.g.
    ``visit_Name``. Refer to the ast module's documentation.

    Attributes:
        locator: The Locator instance to use to translate source
            code locations.
    """

    def __init__(self, locator: Locator) -> None:
        self.locator = locator

    def make_location(self, node) -> Location:
        return self.locator(node)

    def visit(self, node: ast.AST, **kwargs) -> MyiaASTNode:
        loc = self.make_location(node)
        cls = node.__class__.__name__
        try:
            method = getattr(self, 'visit_' + cls)
        except AttributeError:
            raise MyiaSyntaxError(
                loc,
                "Unrecognized Python AST node type: {}".format(cls)
            )
        with About(loc, 'parse'):
            rval = method(node, loc, **kwargs)
        if isinstance(rval, MyiaASTNode):
            rval = rval.at(loc)
        return rval


# Type for AST nodes that can contain a variable name (also str)
InputNode = Union[str, ast.arg, ast.Name]


class Parser(LocVisitor):
    """
    Transform Python's AST into a Myia expression.

    Arguments:
        locator: A Locator to track source code locations.
        global_env: ParseEnv that contains global bindings.
            The Parser will populate it whenever it encounters
            or creates a ``Lambda``.
        macros: A dictionary of macros to customize generation.
            One such macro (aka the only one) is GRAD, see grad.py.
        gen: The GenSym to use to generate new variables.
        dry: Whether to add new symbols to global_env or not.
        pull_free_variables: When free variables are encountered
            and this flag is set, new variables will be created,
            and a mapping will be created in self.free_variables.
            This is used to facilitate creating closures.
        top_level: Whether this is a top level function or not.
            If it isn't top level, its name will be mangled so
            that it doesn't conflict with an existing one in
            global_env.
        return_error: A message string for a MyiaSyntaxError if
            a return statement is encountered (we forbid return
            in some situations).
        dest: Symbol in which the result of the parsing will be
            put. Used to generate derived symbols, for e.g.
            condition/while Lambdas.

    Fields:
        vtrack: VariableChecker to track the mapping from Python
            variables to Symbols. Normally, this is handled
            automatically.
        free_variables: Associates variable names to fresh Symbols
            for each free variable encountered in the expression.
            These are normally used to populate the argument list
            of a Lambda to create a Closure.
        local_assignments: Variables this expression sets.
        returns: Whether this expression returns or not.
    """

    def __init__(self,
                 locator: Locator,
                 global_env: ParseEnv = None,
                 macros: Dict[str, Callable[..., MyiaASTNode]] = None,
                 gen: GenSym = None,
                 dry: bool = None,
                 pull_free_variables: bool = False,
                 top_level: bool = False,
                 return_error: str = None,
                 dest: Symbol = None) \
            -> None:

        super().__init__(locator)
        self.locator = locator
        self.global_env = global_env
        self.gen = gen or GenSym()
        self.dry = dry
        self.pull_free_variables = pull_free_variables
        self.top_level = top_level
        self.macros = macros or {}
        self.return_error = return_error
        self.dest = dest or self.gen(LBDA)

        self.vtrack = VariableTracker()
        self.free_variables: Dict[str, Symbol] = OrderedDict()
        self.local_assignments: Dict[str, bool] = OrderedDict()
        self.returns = False

    def sub_parser(self, **kw) -> 'Parser':
        """
        Return a new Parser derived from this one. Variables
        in the current expression will be free variables in
        the sub_parser. Keyword arguments are passed through
        and have priority over inherited ones.
        """
        dflts = dict(locator=self.locator,
                     global_env=self.global_env,
                     dry=self.dry,
                     return_error=self.return_error,
                     macros=self.macros)
        kw = {**dflts, **kw}
        p = Parser(**kw)
        p.vtrack = VariableTracker(self.vtrack)
        return p

    def base_name(self, input: InputNode) -> str:
        """
        Returns the name of the variable represented by
        this node.
        """
        if isinstance(input, str):
            base_name = input
        elif isinstance(input, ast.arg):
            base_name = input.arg
        elif isinstance(input, ast.Name):
            base_name = input.id
        return base_name

    def reg_lambda(self,
                   ref: Symbol,
                   args: List[Symbol],
                   body: MyiaASTNode,
                   loc: Location) -> Symbol:
        """
        Create a Lambda with the given args and body, and
        associate it to ``ref`` in the global_env (unless
        dry is true). Return ``ref``.
        """
        l = Lambda(args, body, self.gen).at(loc)
        l.ref = ref
        l.global_env = self.global_env
        if not self.dry:
            self.global_env[ref] = l
        return ref.at(loc)

    def new_variable(self, input: InputNode) -> Symbol:
        """
        Create a fresh variable with the same name as the
        given node, but distinct from previous variables
        with the same name. Associate the name to the new
        variable in vtrack. Return the new variable.
        """
        base_name = self.base_name(input)
        loc = self.make_location(input)
        sym = self.gen(base_name).at(loc)
        self.vtrack[base_name] = sym
        return sym

    def declare_new_variable(self, base_name: str):
        sym = self.new_variable(base_name)
        self.local_assignments[base_name] = True
        return sym

    def make_assign(self,
                    base_name: str,
                    value: MyiaASTNode,
                    location: Location = None) -> _Assign:
        """
        Helper function for when a variable with the name
        base_name is set to the given value. Add the name
        to local_assignments. Return an _Assign instance
        that will be made into a Let later on.
        """
        sym = self.declare_new_variable(base_name)
        return _Assign(sym, value, location)

    def multi_assign(self,
                     ass: List[str],
                     expr: MyiaASTNode) -> MyiaASTNode:
        """
        Create an assignment to a Tuple of variables.
        """
        lhs = Tuple(self.declare_new_variable(var) for var in ass)
        return _Assign(lhs, expr, None)

    def prepare_closure(self,
                        variable: str = None,
                        ref: Symbol = None) -> 'Parser':
        """
        In preparation for generating a Closure, given the name
        of a variable and a Symbol that will be the real reference,
        create a Parser for the Closure's body that will create
        new variables for each free variable it encounters.
        """
        if ref is None:
            ref = self.global_env.gen(variable or LBDA)
        p = self.sub_parser(pull_free_variables=True, dest=ref)
        if variable is not None:
            p.vtrack[variable] = ref
        return p

    def construct_closure(self,
                          p: 'Parser',
                          args: List[Symbol],
                          body: MyiaASTNode,
                          loc: Location) -> Union[Closure, Symbol]:
        """
        Construct a Closure as such:

        * Extract the body's free variables. They will be the first
          arguments to the function we close on.
        * Resolve the free variables in the current environment. They
          will be given to Closure so that they can be stored.
        * Concatenate the free variables to the rest of the argument
          variables.
        * Create the Lambda, and the Closure if there is at least one
          free variable.

        Arguments:
            p: The Parser returned by a call to ``prepare_closure``.
            args: The function's remaining argument variables.
            body: The body of the function.
            loc: The location in the code.
        """
        clos_args = list(p.free_variables.values())
        clos_values = [self.visit_variable(v) for v in p.free_variables]
        lbda = self.reg_lambda(p.dest, clos_args + args, body, loc=loc)
        rval: Any
        if len(clos_args) > 0:
            return Closure(lbda, clos_values).at(loc)
        else:
            return lbda

    def make_closure(self,
                     args: List[InputNode],
                     body: Union[Callable, ast.AST, List[ast.stmt]],
                     loc: Location = None,
                     variable: str = "#lambda",
                     ref: Symbol = None) \
            -> Union[Closure, Symbol]:
        """
        Given the Python AST for a list of arguments and the body,
        make the appropriate closure. This combines
        ``prepare_closure``, visiting the body, and
        ``construct_closure``.

        Arguments:
            args: The function's arguments.
            body: The function's body, or a function that will be
                passed a fresh Parser and must return a node.
            loc: Location in source code.
            variable: The name of the function.
            ref: The Symbol for the unique, global reference to the
                function.
        """
        p = self.prepare_closure(variable, ref)
        sargs = [p.new_variable(i) for i in args]
        if callable(body):
            fbody = body(p)
        elif isinstance(body, list):
            fbody = p.visit_body(body)
        else:
            fbody = p.visit(body)
        return self.construct_closure(p, sargs, fbody, loc)

    ###################
    # Visitor helpers #
    ###################

    def body_wrapper(self,
                     stmts: List[ast.stmt]) \
            -> Callable[[Optional[MyiaASTNode]], MyiaASTNode]:
        """
        Process a list of statements as found in the body of a
        Python function, loop or conditional. Produce a Let node.

        Essentially, each statement is categorized as either an
        assignment or an expression. ``body_wrapper`` will identify
        these groups and will appropriately create nested Lets.

        The value of the Let expression can either be a ``return``
        statement, or a tuple of variables that we want to obtain
        because e.g. we need to communicate them outside the body
        of if or while. This is handled by returning a callback:

        Returns:
            A callback that can be passed either a node to return,
            or None:

            * Pass a value to make the innermost Let expression
              return that value.
            * Pass None to signify that the last statement should
              be returned (and raise an error if there is no return
              statement.)
        """
        results: List[MyiaASTNode] = []
        for stmt in stmts:
            ret = self.returns
            r = self.visit(stmt)
            if ret:
                raise MyiaSyntaxError(
                    r.location,
                    "There should be no statements after return."
                )
            if isinstance(r, Begin):
                results += r.stmts
            else:
                results.append(r)

        # We group _Assign nodes together (under the label True),
        # and anything else together (under the label False).
        # The order is preserved by this transformation.
        groups = group_contiguous(results, lambda x: isinstance(x, _Assign))

        def helper(groups, result=None):
            (isass, grp), *rest = groups
            if isass:
                # This is a batch of _Assign nodes, we make a Let.
                bindings = tuple((a.varname, a.value) for a in grp)
                if len(rest) == 0:
                    # There are no more groups to process
                    if result is None:
                        # But we have nothing to return!
                        raise MyiaSyntaxError(
                            grp[-1].location,
                            "Missing return statement."
                        )
                    else:
                        # Return what was given in the callback
                        return Let(bindings, result)
                # Process the rest of the groups
                return Let(bindings, helper(rest, result))
            elif len(rest) == 0:
                # A batch of normal nodes, and it's the last batch.
                if result is not None:
                    # If we want to force a particular result, we
                    # simply add it at the end.
                    grp.append(result)
                if len(grp) == 1:
                    return grp[0]
                else:
                    return Begin(grp)
            else:
                # A batch of normal nodes, we must process the rest.
                return Begin(grp + [helper(rest, result)])

        return lambda v: helper(groups, v)

    def visit_body(self, stmts: List[ast.stmt]) -> MyiaASTNode:
        """
        Visit a list of Python statements.
        """
        # Shortcut for telling body_wrapper to return the block's
        # normal value.
        return self.body_wrapper(stmts)(None)

    def visit_variable(self, name: str, loc: Location = None) -> Symbol:
        """
        Declare that the code we are parsing needs to access the given
        variable, and return the appropriate Symbol associated to that
        variable name.

        This means:

        * If this is a free variable and we want to "pull" free variables,
          create a fresh Symbol and store it in ``free_variables`` instead
          of using the one that already exists. The caller will know what
          to do with them.
        * If the variable is declared neither here nor in the parent, then
          we simply assume this is a global variable and we return a
          Symbol that lives in the global namespace (which might only be
          defined in the future).
        """
        try:
            free, v = self.vtrack.get_free(name)
            if free:
                if self.pull_free_variables:
                    v = self.new_variable(name)
                # v = v.at(loc)
                self.free_variables[name] = v
            return v
        except NameError as e:
            return Symbol(name, namespace='global')

    #################################
    # Visitors for Python AST nodes #
    #################################

    def visit_Assign(self, node: ast.Assign, loc: Location) -> _Assign:
        targ, = node.targets
        if isinstance(targ, ast.Tuple):
            # UNSUPPORTED: x, y = value

            def argtup(targ):
                if isinstance(targ, ast.Tuple):
                    return Tuple(argtup(v) for v in targ.elts)
                elif isinstance(targ, ast.Name):
                    return self.declare_new_variable(targ.id)
                else:
                    raise MyiaSyntaxError(
                        loc,
                        "Deconstructing assignment may only contain"
                        " names and tuples."
                    )

            val = self.visit(node.value)
            return _Assign(argtup(targ), val, location=loc)

        elif isinstance(targ, ast.Subscript):
            if isinstance(targ.value, ast.Name):
                # CASE: x[y] = value
                val = self.visit(node.value)
                slice = Apply(builtins.setslice,
                              self.visit(targ.value),
                              self.visit(targ.slice), val)
                return self.make_assign(targ.value.id, slice, loc)
            else:
                # UNSUPPORTED: f()[x] = value
                raise MyiaSyntaxError(
                    loc,
                    "You can only set a slice on a variable."
                )

        elif isinstance(targ, ast.Name):
            # CASE: x = y
            val = self.visit(node.value)
            return self.make_assign(targ.id, val, loc)

        else:
            # UNSUPPORTED: x.attr = y
            # and others, probably
            raise MyiaSyntaxError(loc, f'Unsupported targ for Assign: {targ}')

    def visit_Attribute(self, node: ast.Attribute, loc: Location) -> Apply:
        # CASE: x.attr
        return Apply(builtins.getattr.at(loc),
                     self.visit(node.value),
                     Value(node.attr).at(loc)).at(loc)

    def visit_AugAssign(self, node: ast.AugAssign, loc: Location) -> _Assign:
        targ = node.target
        if isinstance(targ, ast.Name):
            # CASE: x += y
            aug = self.visit(node.value)
            op = get_operator(node.op).at(loc)
            self.visit_variable(targ.id)
            with About(self.locator(targ), 'parse'):
                prev = self.vtrack.get(targ.id, False)
            val = Apply(op, prev, aug, location=loc)
            return self.make_assign(targ.id, val, loc)
        else:
            # UNSUPPORTED: x[y] += z
            # UNSUPPORTED: x.attr += z
            raise MyiaSyntaxError(
                loc,
                "Augmented assignment to subscripts or "
                "slices is not supported."
            )

    def visit_BinOp(self, node: ast.BinOp, loc: Location) -> Apply:
        # CASE: a + b, a - b, etc.
        op = get_operator(node.op).at(loc)
        l = self.visit(node.left)
        r = self.visit(node.right)
        return Apply(op, l, r, location=loc)

    # def visit_BoolOp(self, node: ast.BoolOp, loc: Location) -> If:
    #     raise MyiaSyntaxError(loc, 'Boolean expressions are not supported.')
    #     left, right = node.values
    #     if isinstance(node.op, ast.And):
    #         return If(self.visit(left), self.visit(right), Value(False))
    #     elif isinstance(node.op, ast.Or):
    #         return If(self.visit(left), Value(True), self.visit(right))
    #     else:
    #         raise MyiaSyntaxError(loc, f"Unknown operator: {node.op}"

    def visit_Call(self, node: ast.Call, loc: Location) -> MyiaASTNode:
        # CASE: f(x, y)
        if len(node.keywords) > 0:
            # UNSUPPORTED: f(x = y), f(**xs)
            raise MyiaSyntaxError(loc, "Keyword arguments are not allowed.")
        for arg in node.args:
            # UNSUPPORTED: f(*xs)
            if isinstance(arg, ast.Starred):
                raise MyiaSyntaxError(self.locator(arg),
                                      "*args are not allowed.")
        args = [self.visit(arg) for arg in node.args]
        if isinstance(node.func, ast.Name) and node.func.id in self.macros:
            return self.macros[node.func.id](*args)
        func = self.visit(node.func)
        return Apply(func, *args, location=loc)

    def visit_Compare(self, node: ast.Compare, loc: Location) -> Apply:
        # CASE: x < y, x == y, etc.
        ops = [get_operator(op) for op in node.ops]
        if len(ops) == 1:
            l = self.visit(node.left)
            cmp = self.visit(node.comparators[0])
            return Apply(ops[0], l, cmp)
        else:
            # UNSUPPORTED: x < y < z
            raise MyiaSyntaxError(
                loc,
                "Comparisons must have a maximum of two operands"
            )

    def visit_Expr(self,
                   node: ast.Expr,
                   loc: Location,
                   allow_decorator='dummy_parameter') -> MyiaASTNode:
        return self.visit(node.value)

    def visit_ExtSlice(self, node: ast.ExtSlice, loc: Location) -> Tuple:
        # CASE: x[a, b, c:d]
        #         ^^^^^^^^^
        return Tuple(self.visit(v) for v in node.dims).at(loc)

    # def visit_For(self, node, loc): # TODO

    def visit_FunctionDef(self,
                          node: ast.FunctionDef,
                          loc: Location,
                          allow_decorator=False) -> _Assign:
        # CASE: def f(x, y): ...
        if node.args.vararg or node.args.kwarg:
            # UNSUPPORTED: def f(x, *y)
            raise MyiaSyntaxError(loc, "Varargs are not allowed.")
        if node.args.kwonlyargs:
            # UNSUPPORTED: def f(x, *, y)
            raise MyiaSyntaxError(
                loc,
                "Keyword-only arguments are not allowed."
            )
        if node.args.defaults or node.args.kw_defaults:
            # UNSUPPORTED: def f(x=dflt)
            raise MyiaSyntaxError(loc, "Default arguments are not allowed.")
        if not allow_decorator and len(node.decorator_list) > 0:
            # UNSUPPORTED: @deco def f(x, y)
            raise MyiaSyntaxError(loc, "Functions should not have decorators.")

        # Global handle for the function
        lbl = node.name if self.top_level else f'{HIDGLOB}{node.name}'
        ref = self.global_env.gen(lbl)

        # Local handle
        sym = self.new_variable(node.name)

        clos = self.make_closure([arg for arg in node.args.args],
                                 node.body,
                                 loc=loc,
                                 variable=node.name,
                                 ref=ref)
        return _Assign(sym, clos, loc)

    def visit_If(self, node: ast.If, loc: Location) -> MyiaASTNode:
        """
        Compile If statement as follows:

        * There must be a then branch *and* an else branch.
        * Both branches become functions. If the parent function is f:
          * The then branch will be a function named ✓f
          * The else branch will be a function named ✗f
          * Nested ifs may produce functions like ✓✓✗f. The outermost
            condition is the closest one to the function name, so you
            should read it right to left.
        * We make a call to ``switch`` on the condition which returns
          either ✓f or ✗f.
        * The result is immediately called with no arguments.

        If an if statement contains assignments to variables:

        * Both branches must assign to the exact same variables.
        * ✓f and ✗f will return a Tuple of the variables' new values.
        * The tuple will be used to set them in the current scope.
        """

        # CASE: if cond: then_branch else: else_branch

        if node.orelse == []:
            # UNSUPPORTED: if cond: then_branch # no else!
            raise MyiaSyntaxError(
                loc,
                "All if statements must be associated to an else statement."
            )

        # Prepare the then branch
        p1 = self.prepare_closure(ref=self.global_env.gen(self.dest, THEN))
        body = p1.body_wrapper(node.body)

        # Prepare the else branch
        p2 = self.prepare_closure(ref=self.global_env.gen(self.dest, ELSE))
        orelse = p2.body_wrapper(node.orelse)

        if p1.returns != p2.returns:
            raise MyiaSyntaxError(
                loc,
                "Either none or all branches of an if statement must return "
                "a value."
            )
        if set(p1.local_assignments) != set(p2.local_assignments):
            raise MyiaSyntaxError(
                loc,
                "All branches of an if statement must assign to the same set "
                " of variables.\nTrue branch sets: {}\nElse branch sets: {}"
                .format(
                    " ".join(sorted(p1.local_assignments)),
                    " ".join(sorted(p2.local_assignments))
                )
            )

        def mkapply(then_finalize, else_finalize) -> Apply:
            then_body = body(then_finalize)
            then_branch = self.construct_closure(
                p1, [], then_body, loc=then_body.location
            )
            else_body = orelse(else_finalize)
            else_branch = self.construct_closure(
                p2, [], else_body, loc=else_body.location
            )
            # In a nutshell, we are doing something similar to this:
            # if test: x; else: y;
            # ==> [(lambda: y), (lambda: x)][test]()
            return Apply(Apply(builtins.switch,
                               self.visit(node.test),
                               then_branch,
                               else_branch,
                               location=loc), location=loc)

        if p1.returns:
            self.returns = True
            return mkapply(None, None)

        else:
            ass = list(p1.local_assignments)

            if len(ass) == 1:
                # Special case when only one variable is set.
                a, = ass
                app = mkapply(p1.vtrack[a], p2.vtrack[a])
                return self.make_assign(a, app, None)

            else:
                app = mkapply(
                    Tuple(p1.vtrack[v] for v in ass),
                    Tuple(p2.vtrack[v] for v in ass)
                )
                return self.multi_assign(ass, app)

    # def visit_IfExp(self, node: ast.IfExp, loc: Location) -> If:
    #     raise MyiaSyntaxError(loc, 'If expressions are not supported.')
    #     return If(self.visit(node.test),
    #               self.visit(node.body),
    #               self.visit(node.orelse),
    #               location=loc)

    def visit_Index(self, node: ast.Index, loc: Location) -> MyiaASTNode:
        # CASE: x[y]
        #         ^
        return self.visit(node.value)

    def visit_Lambda(self, node: ast.Lambda, loc: Location) \
            -> Union[Closure, Symbol]:
        # CASE: lambda x, y: z
        return self.make_closure([a for a in node.args.args],
                                 node.body, loc=loc)

    # def visit_ListComp(self, node: ast.ListComp, loc: Location) \
    #         -> MyiaASTNode:

    #     raise MyiaSyntaxError(loc, 'List comprehensions not supported.')

    #     if len(node.generators) > 1:
    #         raise MyiaSyntaxError(
    #             loc,
    #             "List comprehensions can only iterate over a single target"
    #         )

    #     gen = node.generators[0]
    #     if not isinstance(gen.target, ast.Name):
    #         t = gen.target
    #         raise MyiaSyntaxError(
    #             loc,
    #             f'List comprehension target must be a Name, not {t}'
    #         )

    #     arg: MyiaASTNode = None
    #     if len(gen.ifs) > 0:
    #         test1, *others = reversed(gen.ifs)

    #         def mkcond(p):
    #             cond = p.visit(test1)
    #             for test in others:
    #                 cond = If(p.visit(test), cond, Value(False))
    #             return cond

    #         arg = Apply(builtins.filter,
    #                     self.make_closure([gen.target], mkcond,
    #                                       loc=loc, label="#filtercmp"),
    #                     self.visit(gen.iter))
    #     else:
    #         arg = self.visit(gen.iter)

    #     lbda = self.make_closure(
    #         [gen.target],
    #         node.elt,
    #         loc=loc,
    #         label="#listcmp"
    #     )

    #     return Apply(builtins.map, lbda, arg, location=loc)

    def visit_Module(self, node, loc, allow_decorator=False) \
            -> List[MyiaASTNode]:
        # This is usually the outermost node, we don't really care
        # about it.
        return [self.visit(stmt, allow_decorator=allow_decorator)
                for stmt in node.body]

    def visit_Name(self, node: ast.Name, loc: Location) -> Symbol:
        # CASE: x
        # (A variable name.)
        v = self.visit_variable(node.id, loc)
        v.about = About(loc, 'parse')
        return v

    def visit_NameConstant(self,
                           node: ast.NameConstant,
                           loc: Location) -> Value:
        # CASE: True, False, None... is that it?
        return Value(node.value)

    def visit_Num(self, node: ast.Num, loc: Location) -> Value:
        # CASE: 1, 2, 3.45
        return Value(node.n)

    def visit_Return(self, node: ast.Return, loc: Location) -> MyiaASTNode:
        # CASE: return x
        if self.return_error:
            # In some contexts, e.g. while loops, we ban returning
            # values. This will be relaxed eventually.
            raise MyiaSyntaxError(loc, self.return_error)
        self.returns = True
        return self.visit(node.value).at(loc)

    def visit_Slice(self, node: ast.Slice, loc: Location) -> Apply:
        # CASE: return x[y:z]
        #                ^^^
        return Apply(Symbol('slice'),
                     self.visit(node.lower) if node.lower else Value(0),
                     self.visit(node.upper) if node.upper else Value(None),
                     self.visit(node.step) if node.step else Value(1))

    def visit_Str(self, node: ast.Str, loc: Location) -> Value:
        # CASE: "abc", 'defg'
        return Value(node.s)

    def visit_Tuple(self, node: ast.Tuple, loc: Location) -> Tuple:
        # CASE: (x, y, z)
        return Tuple(self.visit(v) for v in node.elts).at(loc)

    def visit_Subscript(self, node: ast.Subscript, loc: Location) -> Apply:
        # CASE: x[y], x[y, z], etc.
        # TODO: test this
        return Apply(builtins.index,
                     self.visit(node.value),
                     self.visit(node.slice),
                     location=loc)

    def visit_UnaryOp(self, node: ast.UnaryOp, loc: Location) -> Apply:
        # CASE: -x, +x, ~x
        op = get_operator(node.op).at(loc)
        return Apply(op, self.visit(node.operand), location=loc)

    # def explore_vars(self, *exprs, return_error=None):
    #     testp = self.sub_parser(global_env=ParseEnv(), dry=True)
    #     testp.return_error = return_error

    #     for expr in exprs:
    #         if isinstance(expr, list):
    #             testp.body_wrapper(expr)
    #         else:
    #             testp.visit(expr)

    #     invars = testp.free_variables
    #     invars.update(testp.local_assignments)
    #     return {
    #         'in': list(invars),
    #         'out': list(testp.local_assignments)
    #     }

    def visit_While(self, node: ast.While, loc: Location) -> MyiaASTNode:
        """
        A while loop is compiled into two functions. The first function
        tests the condition and either calls the second function, which
        contains the loop body, or returns a tuple of the variables that
        were set in the loop. Assuming the outer function is called f:

        * The test function is called ⤾f. Remember it by the fact it's
          only half a loop. We're only trying to figure out whether to
          perform a loop iteration or not.
        * The loop body is called ⥁f. Remember it by the fact it's a
          complete loop, so at the end of it we performed one full
          iteration.
        * ⤾f calls ⥁f if the test is true. ⥁f calls ⤾f once it has
          finished. Both are tail calls.

        There are two functions because we must be able to execute the
        test without executing the loop body. ⥁f is a sort of shorthand
        for ✓⤾f, basically.
        """

        assert self.dest
        wsym = self.global_env.gen(self.dest, WTEST)
        wbsym = self.global_env.gen(self.dest, WLOOP)

        # We visit the body once to get the free variables
        testp = self.sub_parser(global_env=ParseEnv(), dry=True)
        testp.return_error = 'Cannot return in while loops.'
        testp.visit(node.test)
        testp.body_wrapper(node.body)
        in_vars = testp.free_variables
        in_vars.update(testp.local_assignments)  # type: ignore
        in_vars = list(in_vars)  # type: ignore
        out_vars = list(testp.local_assignments)

        # We visit once more, this time adding the free vars as parameters
        p = self.sub_parser(dest=wbsym)
        in_syms = [p.new_variable(v) for v in in_vars]

        # Have to execute this before the body in order to get the right
        # symbols, otherwise they will be shadowed. If I recall correctly,
        # that's why we need to revisit instead of just using testp.
        initial_values = [p.vtrack[v] for v in out_vars]
        test = p.visit(node.test)
        body = p.body_wrapper(node.body)

        loop_args = in_syms
        loop_body = body(Apply(wsym, *[p.vtrack[v] for v in in_vars])).at(loc)
        loop_fn = self.reg_lambda(wbsym, loop_args, loop_body,
                                  loc=loop_body.location)
        outer_body = Apply(Apply(
            builtins.switch,
            test,
            Closure(loop_fn, in_syms),
            # Closure on identity is a trick to create a thunk
            # that returns a pre-defined value (we need this
            # parameter to be a function with no arguments in
            # order to match the signature of the other branch).
            Closure(builtins.identity, (Tuple(initial_values),))
        ))

        self.reg_lambda(wsym, in_syms, outer_body, loc=loc)

        # We immediately apply our shiny new function to start
        # the loop rolling.
        val = Apply(wsym, *[self.visit_variable(v) for v in in_vars])

        # We get back the variables the while loop sets.
        return self.multi_assign(out_vars, val)


_global_envs: Dict[str, ParseEnv] = {}


def get_global_parse_env(url) -> ParseEnv:
    namespace = f'global'  # :{url}'
    env = ParseEnv(namespace=namespace, url=url)
    return _global_envs.setdefault(url, env)


def parse_source(url: str,
                 line: int,
                 src: str,
                 **kw) -> TupleT[Symbol, ParseEnv]:
    """
    Parse a source string with Myia.

    Arguments:
        url: The filename from whence the source comes.
        line: The line number at which the source starts.
        src: The source code to parse.
        kw: Keyword arguments passed to Parser.

    Returns:
        A pair:
        * The Symbol reference associated to the parsed
          function.
        * The ParseEnv that contains all the bindings that
          were created as a result of parsing. This includes
          the main function being parsed, but also auxiliary
          functions for loop bodies etc.
        To get the Lambda object that corresponds to the
        given source function, index the ParseEnv with the
        Symbol.
    """
    tree = ast.parse(src)
    p = Parser(Locator(url, line),
               get_global_parse_env(url),
               top_level=True,
               **kw)
    r = p.visit(tree, allow_decorator=True)
    if isinstance(r, list):
        r, = r
    if isinstance(r, _Assign):
        r = r.value
    genv = p.global_env
    assert genv is not None
    assert isinstance(r, Symbol)
    return r, genv


def parse_function(fn, **kw) -> TupleT[Symbol, ParseEnv]:
    """
    Parse a function with Myia.

    Arguments:
        fn: A Python function.
        kw: Keyword arguments passed to Parser.

    Returns:
        See ``parse_source``.
    """
    _, line = inspect.getsourcelines(fn)
    return parse_source(inspect.getfile(fn),
                        line,
                        textwrap.dedent(inspect.getsource(fn)),
                        **kw)


def make_error_function(data):
    def _f(*args, **kwargs):
        raise Exception(
            f"Function {data['name']} is for internal use only."
        )
    _f.data = data
    return _f


def myia(fn):
    # This is probably broken.
    _, genv = parse_function(fn)
    gbindings = genv.bindings
    glob = fn.__globals__
    bindings = {k: make_error_function({"name": k, "ast": v, "globals": glob})
                for k, v in gbindings.items()}
    glob.update(bindings)
    fsym = Symbol(fn.__name__, namespace='global')
    fn.data = bindings[fsym].data
    fn.associates = bindings
    return fn
