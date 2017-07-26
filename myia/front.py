from typing import \
    Dict, Set, List, Tuple as TupleT, \
    cast, Union, Callable, Optional

from myia.ast import \
    MyiaASTNode, \
    Location, Symbol, Value, \
    Let, If, Lambda, Apply, \
    Begin, Tuple, Closure, _Assign, \
    GenSym
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


class Redirect:
    def __init__(self, key: str) -> None:
        self.key = key


class Env:
    def __init__(self,
                 parent: 'Env' = None,
                 namespace: str = None,
                 gen: GenSym = None) -> None:
        self.parent = parent
        self.gen: GenSym = gen or \
            parent.gen if parent else GenSym(namespace or str(uuid()))
        self.bindings: Dict[Union[str, Symbol],
                            Union[MyiaASTNode, Redirect]] = {}

    def get_free(self,
                 name: str) -> TupleT[bool, MyiaASTNode]:
        if name in self.bindings:
            free = False
            result = self.bindings[name]
        elif self.parent is None:
            raise NameError("Undeclared variable: {}".format(name))
        else:
            free = True
            result = self.parent[name]
        # if redirect and isinstance(result, Redirect):
        #     return self.get_free(result.key, True)
        # else:
        #     return (free, result)
        if isinstance(result, Redirect):
            return self.get_free(result.key)
        else:
            return (free, result)

    def update(self, bindings) -> None:
        self.bindings.update(bindings)

    def __getitem__(self, name) -> MyiaASTNode:
        _, x = self.get_free(name)
        return x

    def __setitem__(self, name, value) -> None:
        self.bindings[name] = value


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


class Parser(LocVisitor):

    def __init__(self,
                 parent: Union[Locator, 'Parser'],
                 global_env: Env = None,
                 dry: bool = None,
                 gen: GenSym = None,
                 pull_free_variables: bool = False,
                 top_level: bool = False) -> None:
        self.free_variables: Dict[str, Symbol] = {}
        self.local_assignments: Set[str] = set()
        self.returns = False
        self.pull_free_variables = pull_free_variables
        self.top_level = top_level

        if isinstance(parent, Locator):
            self.parent = None
            self.env = Env(gen=gen)
            # self.globals_accessed: Set[str] = set()
            self.global_env = global_env
            self.return_error: str = None
            self.dry = dry
            super().__init__(parent)
        else:
            self.parent = parent
            self.env = Env(parent.env, gen=gen)
            # self.globals_accessed = parent.globals_accessed
            self.global_env = parent.global_env
            self.return_error: str = parent.return_error
            self.dry = parent.dry if dry is None else dry
            super().__init__(parent.locator)

        self.dest = self.gensym('#lambda')

    def gensym(self, name: str) -> Symbol:
        return self.env.gen.sym(name)

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
        if not self.dry:
            self.global_env[ref] = l
        return ref

    def new_variable(self, input: InputNode) -> Symbol:
        base_name = self.base_name(input)
        loc = self.make_location(input)
        sym = self.gensym(base_name).at(loc)
        label = sym.label
        if isinstance(label, Symbol):
            raise TypeError('Label should be a string.')
        self.env.update({base_name: Redirect(label)})
        # The following statement can override the previous,
        # if sym.label == base_name
        # That is fine and intended.
        self.env.update({label: sym})
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
        p = Parser(self, pull_free_variables=True, gen=GenSym())
        if binding is None:
            binding = (label, self.global_env.gen.sym(label))
        sinputs = [p.new_variable(i) for i in inputs]
        p.dest = binding[1]
        p.env[binding[0]] = binding[1]
        p.env.update({k: v for k, v in zip(inputs, sinputs)})
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
            self.env.gen,
            loc=loc,
            binding=binding).at(loc)
        if len(fargs) > 0:
            return Closure(lbda, [self.env[k] for k in fargnames]).at(loc)
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
        prev = self.env[targ.id]
        val = Apply(op, prev, aug, location=loc)
        return self.make_assign(targ.id, val, loc)

    def visit_BinOp(self, node: ast.BinOp, loc: Location) -> Apply:
        op = get_operator(node.op).at(loc)
        return Apply(
            op,
            self.visit(node.left),
            self.visit(node.right), location=loc)

    def visit_BoolOp(self, node: ast.BoolOp, loc: Location) -> If:
        left, right = node.values
        if isinstance(node.op, ast.And):
            return If(self.visit(left), self.visit(right), Value(False))
        elif isinstance(node.op, ast.Or):
            return If(self.visit(left), Value(True), self.visit(right))
        else:
            raise MyiaSyntaxError(loc, "Unknown operator: {}".format(node.op))

    def visit_Call(self, node: ast.Call, loc: Location) -> Apply:
        if (len(node.keywords) > 0):
            raise MyiaSyntaxError(loc, "Keyword arguments are not allowed.")
        return Apply(
            self.visit(node.func),
            *[self.visit(arg) for arg in node.args],
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
        p1 = Parser(self)
        body = p1.body_wrapper(node.body)
        p2 = Parser(self)
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
        if p1.returns:
            self.returns = True
            return If(self.visit(node.test),
                      body(None),
                      orelse(None),
                      location=loc)
        else:
            ass = list(p1.local_assignments)
            if len(ass) == 1:
                a, = ass
                val = If(self.visit(node.test),
                         body(p1.env[a]),
                         orelse(p2.env[a]),
                         location=loc)
                return self.make_assign(a, val, None)
            else:
                val = If(self.visit(node.test),
                         body(Tuple(p1.env[v] for v in ass)),
                         orelse(Tuple(p2.env[v] for v in ass)),
                         location=loc)
                tmp = self.gensym('#tmp')
                stmts = [_Assign(tmp, val, None)]
                for i, a in enumerate(ass):
                    idx = Apply(builtins.index,
                                tmp,
                                Value(i),
                                cannot_fail=True)
                    stmt = self.make_assign(a, idx)
                    stmts.append(stmt)
                return Begin(cast(List[MyiaASTNode], stmts))

    def visit_IfExp(self, node: ast.IfExp, loc: Location) -> If:
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

    def visit_Name(self, node: ast.Name, loc: Location) -> Symbol:
        try:
            free, v = self.env.get_free(node.id)
            assert isinstance(v, Symbol)
            if free:
                if self.pull_free_variables:
                    v = self.new_variable(node.id)
                v = v.at(loc)
                self.free_variables[node.id] = v
            return v
        except NameError as e:
            # raise MyiaSyntaxError(loc, e.args[0])
            # self.globals_accessed.add(node.id)
            return Symbol(node.id, namespace='global')

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

    def visit_While(self, node: ast.While, loc: Location) -> Begin:
        # fsym = self.global_env.gen.sym('#while')
        assert self.dest
        fsym = self.global_env.gen(self.dest, '↻')

        # We visit the body once to get the free variables
        testp = Parser(self, global_env=Env(), dry=True)
        testp.return_error = "While loops cannot contain return statements."
        testp.visit(node.test)
        testp.body_wrapper(node.body)
        in_vars = list(
            set(testp.free_variables.keys()) | testp.local_assignments
        )
        out_vars = list(testp.local_assignments)

        # We visit once more, this time adding the free vars as parameters
        p = Parser(self, gen=GenSym())
        p.dest = fsym
        in_syms = [p.new_variable(v) for v in in_vars]
        # Have to execute this before the body in order to get the right
        # symbols, otherwise they will be shadowed
        initial_values = [p.env[v] for v in out_vars]
        test = p.visit(node.test)
        body = p.body_wrapper(node.body)

        new_body = If(test,
                      body(Apply(fsym, *[p.env[v] for v in in_vars])).at(loc),
                      Tuple(initial_values))

        if not self.dry:
            self.global_env[fsym] = Lambda(
                in_syms,
                new_body,
                p.env.gen,
                location=loc
            )
        # assert isinstance(fsym.label, str)
        # self.globals_accessed.add(fsym.label)

        tmp = self.gensym('#tmp').at(loc)
        val = Apply(fsym, *[self.env[v] for v in in_vars])
        stmts: List[MyiaASTNode] = [_Assign(tmp, val, None)]
        for i, v in enumerate(out_vars):
            stmt = self.make_assign(v, Apply(builtins.index, tmp, Value(i)))
            stmts.append(stmt)
        return Begin(stmts)


def parse_function(fn):
    return parse_function0(fn)[1]


def parse_function0(fn):
    _, line = inspect.getsourcelines(fn)
    return parse_source(inspect.getfile(fn),
                        line,
                        textwrap.dedent(inspect.getsource(fn)))


_global_envs: Dict[str, Env] = {}


def _get_global_env(url):
    return _global_envs.setdefault(url, Env(namespace='global'))


def parse_source(url, line, src):
    tree = ast.parse(src)
    p = Parser(Locator(url, line), _get_global_env(url), top_level=True)
    r = p.visit(tree, allow_decorator=True)

    if isinstance(r, list):
        r, = r
    if isinstance(r, _Assign):
        r = r.value

    # print(p.global_env.bindings)
    # print(p.globals_accessed)

    # for k, v in p.global_env.bindings.items():
    #     _validate(v)

    return r, p.global_env.bindings


def make_error_function(data):
    def _f(*args, **kwargs):
        raise Exception(
            "Function {} is for internal use only.".format(data["name"])
        )
    _f.data = data
    return _f


def myia(fn):
    data = parse_function(fn)
    glob = fn.__globals__
    bindings = {k: make_error_function({"name": k, "ast": v, "globals": glob})
                for k, v in data.items()}
    glob.update(bindings)
    fsym = Symbol(fn.__name__, namespace='global')
    fn.data = bindings[fsym].data
    fn.associates = bindings
    return fn
