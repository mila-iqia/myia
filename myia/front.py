from typing import \
    Dict, Set, List, Tuple as TupleT, \
    cast, Union, Callable, Optional

from .event import EventDispatcher
from myia.ast import \
    MyiaASTNode, \
    Location, Symbol, Value, \
    Let, If, Lambda, Apply, \
    Begin, Tuple, Closure, _Assign, \
    GenSym, ParseEnv
from myia.util import group_contiguous
from myia.symbols import get_operator, builtins
from uuid import uuid4 as uuid
import ast
import inspect
import textwrap
import sys


class MyiaSyntaxError(Exception):
    def __init__(self, location: Location, message: str) -> None:
        self.location = location
        self.message = message


_prevhook = sys.excepthook


def exception_handler(exception_type, exception, traceback):
    if exception_type == MyiaSyntaxError:
        print(
            "{}: {}".format(exception_type.__name__, exception.message),
            file=sys.stderr
        )
        if exception.location:
            print(exception.location.traceback(), file=sys.stderr)
    else:
        _prevhook(exception_type, exception, traceback)


sys.excepthook = exception_handler


class Locator:
    def __init__(self, url: str, line_offset: int) -> None:
        self.url = url
        self.line_offset = line_offset

    def __call__(self, node: Union[ast.expr, ast.stmt]) -> Location:
        try:
            return Location(
                self.url,
                node.lineno + self.line_offset - 1,
                node.col_offset
            )
        except AttributeError:
            return None


class LocVisitor:
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
        rval = method(node, loc, **kwargs)
        if isinstance(rval, MyiaASTNode):
            rval = rval.at(loc)
        return rval


InputNode = Union[str, ast.arg, ast.Name]


class VariableTracker:
    def __init__(self, parent: 'VariableTracker' = None) -> None:
        self.parent = parent
        self.bindings: Dict[str, Symbol] = {}

    def get_free(self, name: str) -> TupleT[bool, 'Symbol']:
        if name in self.bindings:
            return (False, self.bindings[name])
        elif self.parent is None:
            raise NameError("Undeclared variable: {}".format(name))
        else:
            return (True, self.parent.get_free(name)[1])

    def __getitem__(self, name: str) -> Symbol:
        return self.get_free(name)[1]

    def __setitem__(self, name: str, value: Symbol) -> None:
        self.bindings[name] = value


class Parser(LocVisitor):

    def __init__(self,
                 locator: Locator,
                 global_env: ParseEnv = None,
                 macros: Dict[str, Callable[..., MyiaASTNode]] = None,
                 gen: GenSym = None,
                 dry: bool = None,
                 pull_free_variables: bool = False,
                 top_level: bool = False,
                 return_error: str = None,
                 vtrack: VariableTracker = None) \
            -> None:

        super().__init__(locator)
        self.locator = locator
        self.global_env = global_env
        self.gen = gen or GenSym()
        self.vtrack = vtrack or VariableTracker()
        self.dry = dry
        self.pull_free_variables = pull_free_variables
        self.top_level = top_level
        self.macros = macros or {}

        self.free_variables: Dict[str, Symbol] = {}
        self.local_assignments: Set[str] = set()
        self.returns = False
        self.return_error = return_error
        self.dest = self.gensym('#lambda')

    def sub_parser(self, **kw):
        dflts = dict(locator=self.locator,
                     global_env=self.global_env,
                     gen=self.gen,
                     vtrack=VariableTracker(self.vtrack),
                     dry=self.dry,
                     return_error=self.return_error,
                     macros=self.macros)
        kw = {**dflts, **kw}
        return Parser(**kw)

    def gensym(self, name: str) -> Symbol:
        return self.gen.sym(name)

    def base_name(self, input: InputNode) -> str:
        if isinstance(input, str):
            base_name = input
        elif isinstance(input, ast.arg):
            base_name = input.arg
        elif isinstance(input, ast.Name):
            base_name = input.id
        return base_name

    def reg_lambda(self,
                   args: List[Symbol],
                   body: MyiaASTNode,
                   gen: GenSym,
                   loc: Location = None,
                   label: str = "#lambda",
                   binding: TupleT[str, Symbol] = None) -> Symbol:
        ref = binding[1] if binding else self.global_env.gen.sym(label)
        l = Lambda(args, body, gen).at(loc)
        l.ref = ref
        l.global_env = self.global_env
        if not self.dry:
            self.global_env[ref] = l
        return ref

    def new_variable(self, input: InputNode) -> Symbol:
        base_name = self.base_name(input)
        loc = self.make_location(input)
        sym = self.gensym(base_name).at(loc)
        self.vtrack[base_name] = sym
        return sym

    def make_assign(self,
                    base_name: str,
                    value: MyiaASTNode,
                    location: Location = None) -> _Assign:
        sym = self.new_variable(base_name)
        self.local_assignments.add(base_name)
        return _Assign(sym, value, location)

    def make_closure(self,
                     inputs: List[InputNode],
                     expr: Union[Callable, ast.AST, List[ast.stmt]],
                     loc: Location = None,
                     label: str = "#lambda",
                     binding: TupleT[str, Symbol] = None) \
            -> Union[Closure, Symbol]:
        p = self.sub_parser(pull_free_variables=True, gen=GenSym())
        if binding is None:
            binding = (label, self.global_env.gen.sym(label))
        sinputs = [p.new_variable(i) for i in inputs]
        p.dest = binding[1]
        p.vtrack[binding[0]] = binding[1]
        for k, v in zip(inputs, sinputs):
            p.vtrack[self.base_name(k)] = v
        if callable(expr):
            body = expr(p)
        elif isinstance(expr, list):
            body = p.visit_body(expr)
        else:
            body = p.visit(expr)
        fargnames = list(p.free_variables.keys())
        fargs = [p.free_variables[k] for k in fargnames]
        lbda = self.reg_lambda(
            fargs + sinputs,
            body,
            self.gen,
            loc=loc,
            binding=binding).at(loc)
        if len(fargs) > 0:
            return Closure(lbda, [self.vtrack[k] for k in fargnames]).at(loc)
        else:
            return lbda

    def body_wrapper(self,
                     stmts: List[ast.stmt]) \
            -> Callable[[Optional[MyiaASTNode]], MyiaASTNode]:
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
        groups = group_contiguous(results, lambda x: isinstance(x, _Assign))

        def helper(groups, result=None):
            (isass, grp), *rest = groups
            if isass:
                bindings = tuple((a.varname, a.value) for a in grp)
                if len(rest) == 0:
                    if result is None:
                        raise MyiaSyntaxError(
                            grp[-1].location,
                            "Missing return statement."
                        )
                    else:
                        return Let(bindings, result)
                return Let(bindings, helper(rest, result))
            elif len(rest) == 0:
                if len(grp) == 1:
                    return grp[0]
                else:
                    return Begin(grp)
            else:
                return Begin(grp + [helper(rest, result)])

        return lambda v: helper(groups, v)

    def visit_body(self, stmts: List[ast.stmt]) -> MyiaASTNode:
        return self.body_wrapper(stmts)(None)

    # def visit_arg(self, node, loc):
    #     return Symbol(node.arg, location=loc)

    # def visit_arguments(self, args):
    #     return [self.visit(arg) for arg in args.args]

    def visit_Assign(self, node: ast.Assign, loc: Location) -> _Assign:
        targ, = node.targets
        if isinstance(targ, ast.Tuple):
            raise MyiaSyntaxError(
                loc,
                "Deconstructing assignment is not supported."
            )
        if isinstance(targ, ast.Subscript):
            if not isinstance(targ.value, ast.Name):
                raise MyiaSyntaxError(
                    loc,
                    "You can only set a slice on a variable."
                )

            val = self.visit(node.value)
            slice = Apply(builtins.setslice,
                          self.visit(targ.value),
                          self.visit(targ.slice), val)
            return self.make_assign(targ.value.id, slice, loc)

        elif isinstance(targ, ast.Name):
            val = self.visit(node.value)
            return self.make_assign(targ.id, val, loc)

        else:
            raise MyiaSyntaxError(loc, f'Unsupported targ for Assign: {targ}')

    def visit_Attribute(self, node: ast.Attribute, loc: Location) -> Apply:
        return Apply(builtins.getattr.at(loc),
                     self.visit(node.value),
                     Value(node.attr).at(loc)).at(loc)

    def visit_AugAssign(self, node: ast.AugAssign, loc: Location) -> _Assign:
        targ = node.target
        if not isinstance(targ, ast.Name):
            raise MyiaSyntaxError(
                loc,
                "Augmented assignment to subscripts or "
                "slices is not supported."
            )
        aug = self.visit(node.value)
        op = get_operator(node.op).at(loc)
        self.visit_variable(targ.id)
        prev = self.vtrack[targ.id]
        val = Apply(op, prev, aug, location=loc)
        return self.make_assign(targ.id, val, loc)

    def visit_BinOp(self, node: ast.BinOp, loc: Location) -> Apply:
        op = get_operator(node.op).at(loc)
        return Apply(
            op,
            self.visit(node.left),
            self.visit(node.right), location=loc)

    def visit_BoolOp(self, node: ast.BoolOp, loc: Location) -> If:
        raise MyiaSyntaxError(loc, 'Boolean expressions are not supported.')
        left, right = node.values
        if isinstance(node.op, ast.And):
            return If(self.visit(left), self.visit(right), Value(False))
        elif isinstance(node.op, ast.Or):
            return If(self.visit(left), Value(True), self.visit(right))
        else:
            raise MyiaSyntaxError(loc, "Unknown operator: {}".format(node.op))

    def visit_Call(self, node: ast.Call, loc: Location) -> MyiaASTNode:
        if (len(node.keywords) > 0):
            raise MyiaSyntaxError(loc, "Keyword arguments are not allowed.")
        args = [self.visit(arg) for arg in node.args]
        if isinstance(node.func, ast.Name) and node.func.id in self.macros:
            return self.macros[node.func.id](*args)
        return Apply(
            self.visit(node.func),
            *args,
            # *[self.visit(arg) for arg in node.args],
            location=loc
        )

    def visit_Compare(self, node: ast.Compare, loc: Location) -> Apply:
        ops = [get_operator(op) for op in node.ops]
        if len(ops) == 1:
            return Apply(
                ops[0],
                self.visit(node.left),
                self.visit(node.comparators[0])
            )
        else:
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
        return Tuple(self.visit(v) for v in node.dims).at(loc)

    # def visit_For(self, node, loc): # TODO

    def visit_FunctionDef(self,
                          node: ast.FunctionDef,
                          loc: Location,
                          allow_decorator=False) -> _Assign:
        if node.args.vararg:
            raise MyiaSyntaxError(loc, "Varargs are not allowed.")
        if node.args.kwarg:
            raise MyiaSyntaxError(loc, "Varargs are not allowed.")
        if node.args.kwonlyargs:
            raise MyiaSyntaxError(
                loc,
                "Keyword-only arguments are not allowed."
            )
        if node.args.defaults or node.args.kw_defaults:
            raise MyiaSyntaxError(loc, "Default arguments are not allowed.")
        if not allow_decorator and len(node.decorator_list) > 0:
            raise MyiaSyntaxError(loc, "Functions should not have decorators.")

        lbl = node.name if self.top_level else '#:' + node.name
        binding = (node.name, self.global_env.gen.sym(lbl))

        sym = self.new_variable(node.name)
        clos = self.make_closure([arg for arg in node.args.args],
                                 node.body,
                                 loc=loc,
                                 binding=binding)
        return _Assign(sym, clos, loc)

    def visit_If(self, node: ast.If, loc: Location) \
            -> Union[MyiaASTNode, _Assign]:

        p1 = self.sub_parser(pull_free_variables=True, gen=GenSym())
        p1.dest = self.global_env.gen(self.dest, '✓')
        body = p1.body_wrapper(node.body)

        p2 = self.sub_parser(pull_free_variables=True, gen=GenSym())
        p2.dest = self.global_env.gen(self.dest, '✗')
        orelse = p2.body_wrapper(node.orelse)

        if p1.returns != p2.returns:
            raise MyiaSyntaxError(
                loc,
                "Either none or all branches of an if statement must return "
                "a value."
            )
        if p1.local_assignments != p2.local_assignments:
            raise MyiaSyntaxError(
                loc,
                "All branches of an if statement must assign to the same set "
                " of variables.\nTrue branch sets: {}\nElse branch sets: {}"
                .format(
                    " ".join(sorted(p1.local_assignments)),
                    " ".join(sorted(p2.local_assignments))
                )
            )

        for k in p1.free_variables:
            self.visit_variable(k)
        for k in p2.free_variables:
            self.visit_variable(k)

        then_args = [sym for v, sym in p1.free_variables.items()]
        then_vars = [self.vtrack[v] for v in p1.free_variables]

        else_args = [sym for v, sym in p2.free_variables.items()]
        else_vars = [self.vtrack[v] for v in p2.free_variables]

        def mkapply(then_body, else_body):
            then_fn = self.reg_lambda(
                then_args, then_body, self.gen, None, "#if",
                (None, p1.dest)
            )
            then_branch = Closure(then_fn, then_vars)

            else_fn = self.reg_lambda(
                else_args, else_body, self.gen, None, "#if",
                (None, p2.dest)
            )
            else_branch = Closure(else_fn, else_vars)
            return Apply(Apply(builtins.switch,
                               self.visit(node.test),
                               then_branch,
                               else_branch,
                               location=loc), location=loc)

        if p1.returns:
            self.returns = True
            return mkapply(body(None), orelse(None))

        else:
            ass = list(p1.local_assignments)

            if len(ass) == 1:
                a, = ass
                app = mkapply(body(p1.vtrack[a]), orelse(p2.vtrack[a]))
                return self.make_assign(a, app, None)

            else:
                app = mkapply(
                    body(Tuple(p1.vtrack[v] for v in ass)),
                    orelse(Tuple(p2.vtrack[v] for v in ass))
                )
                tmp = self.gensym('#tmp')
                stmts = [_Assign(tmp, app, None)]
                for i, a in enumerate(ass):
                    idx = Apply(builtins.index,
                                tmp,
                                Value(i),
                                cannot_fail=True)
                    stmt = self.make_assign(a, idx)
                    stmts.append(stmt)
                return Begin(cast(List[MyiaASTNode], stmts))

    def visit_IfExp(self, node: ast.IfExp, loc: Location) -> If:
        raise MyiaSyntaxError(loc, 'If expressions are not supported.')
        return If(self.visit(node.test),
                  self.visit(node.body),
                  self.visit(node.orelse),
                  location=loc)

    def visit_Index(self, node: ast.Index, loc: Location) -> MyiaASTNode:
        return self.visit(node.value)

    def visit_Lambda(self, node: ast.Lambda, loc: Location) \
            -> Union[Closure, Symbol]:
        return self.make_closure([a for a in node.args.args],
                                 node.body, loc=loc).at(loc)

    def visit_ListComp(self, node: ast.ListComp, loc: Location) \
            -> MyiaASTNode:

        raise MyiaSyntaxError(loc, 'List comprehensions not supported.')

        if len(node.generators) > 1:
            raise MyiaSyntaxError(
                loc,
                "List comprehensions can only iterate over a single target"
            )

        gen = node.generators[0]
        if not isinstance(gen.target, ast.Name):
            t = gen.target
            raise MyiaSyntaxError(
                loc,
                f'List comprehension target must be a Name, not {t}'
            )

        arg: MyiaASTNode = None
        if len(gen.ifs) > 0:
            test1, *others = reversed(gen.ifs)

            def mkcond(p):
                cond = p.visit(test1)
                for test in others:
                    cond = If(p.visit(test), cond, Value(False))
                return cond

            arg = Apply(builtins.filter,
                        self.make_closure([gen.target], mkcond,
                                          loc=loc, label="#filtercmp"),
                        self.visit(gen.iter))
        else:
            arg = self.visit(gen.iter)

        lbda = self.make_closure(
            [gen.target],
            node.elt,
            loc=loc,
            label="#listcmp"
        )

        return Apply(builtins.map, lbda, arg, location=loc)

    def visit_Module(self, node, loc, allow_decorator=False):
        return [self.visit(stmt, allow_decorator=allow_decorator)
                for stmt in node.body]

    def visit_variable(self, name: str, loc: Location = None) -> Symbol:
        try:
            # free, v = self.env.get_free(name)
            # assert isinstance(v, Symbol)
            free, v = self.vtrack.get_free(name)
            if free:
                if self.pull_free_variables:
                    v = self.new_variable(name)
                v = v.at(loc)
                self.free_variables[name] = v
            return v
        except NameError as e:
            # raise MyiaSyntaxError(loc, e.args[0])
            # self.globals_accessed.add(name)
            return Symbol(name, namespace='global')

    def visit_Name(self, node: ast.Name, loc: Location) -> Symbol:
        return self.visit_variable(node.id, loc)

    def visit_NameConstant(self,
                           node: ast.NameConstant,
                           loc: Location) -> Value:
        return Value(node.value)

    def visit_Num(self, node: ast.Num, loc: Location) -> Value:
        return Value(node.n)

    def visit_Return(self, node: ast.Return, loc: Location) -> MyiaASTNode:
        if self.return_error:
            raise MyiaSyntaxError(loc, self.return_error)
        self.returns = True
        return self.visit(node.value).at(loc)

    def visit_Slice(self, node: ast.Slice, loc: Location) -> Apply:
        return Apply(Symbol('slice'),
                     self.visit(node.lower) if node.lower else Value(0),
                     self.visit(node.upper) if node.upper else Value(None),
                     self.visit(node.step) if node.step else Value(1))

    def visit_Str(self, node: ast.Str, loc: Location) -> Value:
        return Value(node.s)

    def visit_Tuple(self, node: ast.Tuple, loc: Location) -> Tuple:
        return Tuple(self.visit(v) for v in node.elts).at(loc)

    def visit_Subscript(self, node: ast.Subscript, loc: Location) -> Apply:
        # TODO: test this
        return Apply(builtins.index,
                     self.visit(node.value),
                     self.visit(node.slice),
                     location=loc)

    def visit_UnaryOp(self, node: ast.UnaryOp, loc: Location) -> Apply:
        op = get_operator(node.op).at(loc)
        return Apply(op, self.visit(node.operand), location=loc)

    def explore_vars(self, *exprs, return_error=None):
        testp = self.sub_parser(global_env=ParseEnv(), dry=True)
        testp.return_error = return_error

        for expr in exprs:
            if isinstance(expr, list):
                testp.body_wrapper(expr)
            else:
                testp.visit(expr)

        return {
            'in': list(
                set(testp.free_variables.keys()) | testp.local_assignments
            ),
            'out': list(testp.local_assignments)
        }

    def visit_While(self, node: ast.While, loc: Location) -> Begin:
        assert self.dest
        wsym = self.global_env.gen(self.dest, '↻')
        wbsym = self.global_env.gen(self.dest, '⥁')

        # We visit the body once to get the free variables
        while_vars = self.explore_vars(node.test, node.body)
        in_vars = while_vars['in']
        out_vars = while_vars['out']

        for v in in_vars:
            self.visit_variable(v)

        # We visit once more, this time adding the free vars as parameters
        p = self.sub_parser(gen=GenSym())
        p.dest = wsym
        in_syms = [p.new_variable(v) for v in in_vars]
        # Have to execute this before the body in order to get the right
        # symbols, otherwise they will be shadowed
        initial_values = [p.vtrack[v] for v in out_vars]
        test = p.visit(node.test)
        body = p.body_wrapper(node.body)

        if_args = in_syms
        if_body = body(Apply(wsym, *[p.vtrack[v] for v in in_vars])).at(loc)
        if_fn = self.reg_lambda(
            if_args, if_body, self.gen, None, "#while_if",
            (None, wbsym)
        )
        new_body = Apply(Apply(
            builtins.switch,
            test,
            Closure(if_fn, in_syms),
            Closure(builtins.identity, (Tuple(initial_values),))
        ))

        if not self.dry:
            l = Lambda(
                in_syms,
                new_body,
                p.gen,
                location=loc
            )
            l.ref = wsym
            l.global_env = self.global_env
            self.global_env[wsym] = l
        # assert isinstance(wsym.label, str)
        # self.globals_accessed.add(wsym.label)

        tmp = self.gensym('#tmp').at(loc)
        val = Apply(wsym, *[self.vtrack[v] for v in in_vars])
        stmts: List[MyiaASTNode] = [_Assign(tmp, val, None)]
        for i, v in enumerate(out_vars):
            stmt = self.make_assign(v, Apply(builtins.index, tmp, Value(i)))
            stmts.append(stmt)
        return Begin(stmts)


def parse_function(fn, **kw) -> TupleT[Symbol, ParseEnv]:
    _, line = inspect.getsourcelines(fn)
    return parse_source(inspect.getfile(fn),
                        line,
                        textwrap.dedent(inspect.getsource(fn)),
                        **kw)


_global_envs: Dict[str, ParseEnv] = {}


def get_global_parse_env(url) -> ParseEnv:
    namespace = f'global'  # :{url}'
    env = ParseEnv(namespace=namespace, url=url)
    return _global_envs.setdefault(url, env)


def parse_source(url: str,
                 line: int,
                 src: str,
                 **kw) -> TupleT[Symbol, ParseEnv]:
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


def make_error_function(data):
    def _f(*args, **kwargs):
        raise Exception(
            f"Function {data['name']} is for internal use only."
        )
    _f.data = data
    return _f


def myia(fn):
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
