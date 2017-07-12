
from myia.ast import \
    MyiaASTNode, \
    Location, Symbol, Literal, \
    LetRec, If, Lambda, Apply, Begin, Tuple, Closure
from myia.symbols import get_operator, builtins
from uuid import uuid4 as uuid
import ast
import inspect
import textwrap
import sys


class MyiaSyntaxError(Exception):
    def __init__(self, location, message):
        self.location = location
        self.message = message


_prevhook = sys.excepthook
def exception_handler(exception_type, exception, traceback):
    if (exception_type == MyiaSyntaxError):
        print("{}: {}".format(exception_type.__name__, exception.message), file=sys.stderr)
        if exception.location:
            print(exception.location.traceback(), file=sys.stderr)
    else:
        _prevhook(exception_type, exception, traceback)
sys.excepthook = exception_handler


class Redirect:
    def __init__(self, key):
        self.key = key


class GenSym:
    def __init__(self, namespace):
        self.varcounts = {}
        self.namespace = namespace

    def name(self, name):
        if name in self.varcounts:
            self.varcounts[name] += 1
            return '{}#{}'.format(name, self.varcounts[name])
        else:
            self.varcounts[name] = 0
            return name

    def sym(self, name, namespace=None):
        return Symbol(self.name(name), namespace=namespace or self.namespace)


class Env:
    def __init__(self, parent=None, namespace=None):
        self.parent = parent
        self.gen = parent.gen if parent else GenSym(namespace or str(uuid()))
        self.bindings = {}

    def get_free(self, name, redirect=True):
        if name in self.bindings:
            free = False
            result = self.bindings[name]
        elif self.parent is None:
            raise NameError("Undeclared variable: {}".format(name))
        else:
            free = True
            result = self.parent[name]
        if redirect and isinstance(result, Redirect):
            return self.get_free(result.key, True)
        else:
            return (free, result)

    def update(self, bindings):
        self.bindings.update(bindings)

    def __getitem__(self, name):
        _, x = self.get_free(name)
        return x

    def __setitem__(self, name, value):
        self.bindings[name] = value


class Locator:
    def __init__(self, url, line_offset):
        self.url = url
        self.line_offset = line_offset

    def __call__(self, node):
        try:
            return Location(self.url, node.lineno + self.line_offset - 1, node.col_offset)
        except AttributeError:
            return None


class LocVisitor:
    def __init__(self, locator):
        self.locator = locator

    def make_location(self, node):
        return self.locator(node)

    def visit(self, node, **kwargs):
        loc = self.make_location(node)
        cls = node.__class__.__name__
        try:
            method = getattr(self, 'visit_' + cls)
        except AttributeError:
            raise MyiaSyntaxError(loc,
                                  "Unrecognized Python AST node type: {}".format(cls))
        rval = method(node, loc, **kwargs)
        if isinstance(rval, MyiaASTNode):
            rval = rval.at(loc)
        return rval


class _Assign:
    def __init__(self, varname, value, location):
        self.varname = varname
        self.value = value
        self.location = location


_c = object()
def group(arr, classify):
    current_c = _c
    results = []
    current = []
    for a in arr:
        c = classify(a)
        if current_c == c:
            current.append(a)
        else:
            if current_c is not _c:
                results.append((current_c, current))
            current_c = c
            current = [a]
    if current_c is not _c:
        results.append((current_c, current))
    return results


class Parser(LocVisitor):

    def __init__(self, parent, global_env=None, dry=None,
                 pull_free_variables=False,
                 top_level=False):
        self.free_variables = {}
        self.local_assignments = set()
        self.returns = False
        self.pull_free_variables = pull_free_variables
        self.top_level = top_level
        
        if isinstance(parent, Locator):
            self.parent = None
            self.env = Env()
            self.globals_accessed = set()
            self.global_env = global_env
            self.return_error = None
            self.dry = dry
            super().__init__(parent)
        else:
            self.parent = parent
            self.env = Env(parent.env)
            self.globals_accessed = parent.globals_accessed
            self.global_env = parent.global_env
            self.return_error = parent.return_error
            self.dry = parent.dry if dry is None else dry
            super().__init__(parent.locator)

    def gensym(self, name):
        return self.env.gen.sym(name)

    def base_name(self, input):
        if isinstance(input, str):
            base_name = input
        elif isinstance(input, ast.arg):
            base_name = input.arg
        elif isinstance(input, ast.Name):
            base_name = input.id
        return base_name

    def reg_lambda(self, args, body, loc=None, label="#lambda", binding=None):
        ref = binding[1] if binding else self.global_env.gen.sym(label)
        l = Lambda(ref.label, args, body).at(loc)
        if not self.dry:
            self.global_env[ref.label] = l
        return ref

    def new_variable(self, input):
        base_name = self.base_name(input)
        loc = self.make_location(input)
        sym = self.gensym(base_name).at(loc)
        self.env.update({base_name: Redirect(sym.label)})
        # The following statement can override the previous, if sym.label == base_name
        # That is fine and intended.
        self.env.update({sym.label: sym})
        return sym

    def make_assign(self, base_name, value, location=None):
        sym = self.new_variable(base_name)
        self.local_assignments.add(base_name)
        return _Assign(sym, value, location)

    def make_closure(self, inputs, expr, loc=None, label="#lambda", binding=None):
        p = Parser(self, pull_free_variables=True)
        binding = binding if binding else (label, self.global_env.gen.sym(label))
        sinputs = [p.new_variable(i) for i in inputs]
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
        lbda = self.reg_lambda(fargs + sinputs, body, loc=loc, binding=binding).at(loc)
        if len(fargs) > 0:
            return Closure(lbda, [self.env[k] for k in fargnames]).at(loc)
        else:
            return lbda

    def visit_body(self, stmts, return_wrapper=False):
        results = []
        for stmt in stmts:
            ret = self.returns
            r = self.visit(stmt)
            if ret:
                raise MyiaSyntaxError(r.location,
                                      "There should be no statements after return.")
            if isinstance(r, Begin):
                results += r.stmts
            else:
                results.append(r)
        groups = group(results, lambda x: isinstance(x, _Assign))
        def helper(groups, result=None):
            (isass, grp), *rest = groups
            if isass:
                bindings = tuple((a.varname, a.value) for a in grp)
                if len(rest) == 0:
                    if result is None:
                        raise MyiaSyntaxError(grp[-1].location, "Missing return statement.")
                    else:
                        return LetRec(bindings, result)
                return LetRec(bindings, helper(rest, result))
            elif len(rest) == 0:
                if len(grp) == 1:
                    return grp[0]
                else:
                    return Begin(grp)
            else:
                return Begin(grp + [helper(rest, result)])

        if return_wrapper:
            return lambda v: helper(groups, v)
        else:
            return helper(groups)

    # def visit_arg(self, node, loc):
    #     return Symbol(node.arg, location=loc)

    # def visit_arguments(self, args):
    #     return [self.visit(arg) for arg in args.args]

    def visit_Assign(self, node, loc):
        targ, = node.targets
        if isinstance(targ, ast.Tuple):
            raise MyiaSyntaxError(loc, "Deconstructing assignment is not supported.")
        if isinstance(targ, ast.Subscript):
            if not isinstance(targ.value, ast.Name):
                raise MyiaSyntaxError(loc, "You can only set a slice on a variable.")
            print(dir(targ.slice))

            val = self.visit(node.value)
            slice = Apply(builtins.setslice,
                          self.visit(targ.value),
                          self.visit(targ.slice), val)
            return self.make_assign(targ.value.id, slice, loc)
            
        else:
            val = self.visit(node.value)
            return self.make_assign(targ.id, val, loc)

    def visit_Attribute(self, node, loc):
        return Apply(builtins.getattr.at(loc),
                     self.visit(node.value),
                     Literal(node.attr).at(loc)).at(loc)

    def visit_AugAssign(self, node, loc):
        targ = node.target
        if isinstance(targ, ast.Subscript):
            raise MyiaSyntaxError(loc, "Augmented assignment to subscripts or slices is not supported.")
        aug = self.visit(node.value)
        op = get_operator(node.op).at(loc)
        prev = self.env[targ.id]
        val = Apply(op, prev, aug, location=loc)
        return self.make_assign(targ.id, val, loc)

    def visit_BinOp(self, node, loc):
        op = get_operator(node.op).at(loc)
        return Apply(op, self.visit(node.left), self.visit(node.right), location=loc)

    def visit_BoolOp(self, node, loc):
        left, right = node.values
        if isinstance(node.op, ast.And):
            return If(self.visit(left), self.visit(right), Literal(False))
        elif isinstance(node.op, ast.Or):
            return If(self.visit(left), Literal(True), self.visit(right))
        else:
            raise MyiaSyntaxError(loc, "Unknown operator: {}".format(node.op))

    def visit_Call(self, node, loc):
        if (len(node.keywords) > 0):
            raise MyiaSyntaxError(loc, "Keyword arguments are not allowed.")
        return Apply(self.visit(node.func),
                     *[self.visit(arg) for arg in node.args],
                     location=loc)


    def visit_Compare(self, node, loc):
        ops = [get_operator(op) for op in node.ops]
        if len(ops) == 1:
            return Apply(ops[0], self.visit(node.left), self.visit(node.comparators[0]))
        else:
            raise MyiaSyntaxError(loc,
                                  "Comparisons must have a maximum of two operands")

    def visit_Expr(self, node, loc, allow_decorator='this is a dummy_parameter'):
        return self.visit(node.value)

    def visit_ExtSlice(self, node, loc):
        return Tuple(self.visit(v) for v in node.dims).at(loc)

    # def visit_For(self, node, loc): # TODO

    def visit_FunctionDef(self, node, loc, allow_decorator=False):
        if node.args.vararg:
            raise MyiaSyntaxError(loc, "Varargs are not allowed.")
        if node.args.kwarg:
            raise MyiaSyntaxError(loc, "Varargs are not allowed.")
        if node.args.kwonlyargs:
            raise MyiaSyntaxError(loc, "Keyword-only arguments are not allowed.")
        if node.args.defaults or node.args.kw_defaults:
            raise MyiaSyntaxError(loc, "Default arguments are not allowed.")
        if not allow_decorator and len(node.decorator_list) > 0:
            raise MyiaSyntaxError(loc, "Functions should not have decorators.")

        lbl = node.name if self.top_level else '#:' + node.name
        binding = (node.name, self.global_env.gen.sym(lbl))

        return self.make_closure([arg for arg in node.args.args],
                                 node.body,
                                 loc=loc,
                                 binding=binding)

    def visit_If(self, node, loc):
        p1 = Parser(self)
        body = p1.visit_body(node.body, True)
        p2 = Parser(self)
        orelse = p2.visit_body(node.orelse, True)
        if p1.returns != p2.returns:
            raise MyiaSyntaxError(loc, "Either none or all branches of an if statement must return a value.")
        if p1.local_assignments != p2.local_assignments:
            raise MyiaSyntaxError(loc, "All branches of an if statement must assign to the same set of variables.\nTrue branch sets: {}\nElse branch sets: {}".format(" ".join(sorted(p1.local_assignments)), " ".join(sorted(p2.local_assignments))))

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
                                Literal(i).at(a.location),
                                cannot_fail=True)
                    stmt = self.make_assign(a, idx)
                    stmts.append(stmt)
                return Begin(stmts)

    def visit_IfExp(self, node, loc):
        return If(self.visit(node.test),
                  self.visit(node.body),
                  self.visit(node.orelse),
                  location=loc)

    def visit_Index(self, node, loc):
        return self.visit(node.value)

    def visit_Lambda(self, node, loc):
        return self.make_closure([a for a in node.args.args],
                                 node.body, loc=loc).at(loc)

    def visit_ListComp(self, node, loc):
        if len(node.generators) > 1:
            raise MyiaSyntaxError(loc,
                "List comprehensions can only iterate over a single target")

        gen = node.generators[0]
        if len(gen.ifs) > 0:
            test1, *others = reversed(gen.ifs)
            def mkcond(p):
                cond = p.visit(test1)
                for test in others:
                    cond = If(p.visit(test), cond, Literal(False))
                return cond
            arg = Apply(builtins.filter,
                        self.make_closure([gen.target], mkcond,
                                          loc=loc, label="#filtercmp"),
                        self.visit(gen.iter))
        else:
            arg = self.visit(gen.iter)

        lbda = self.make_closure([gen.target], node.elt, loc=loc, label="#listcmp")

        return Apply(builtins.map, lbda, arg, location=loc)

    def visit_Module(self, node, loc, allow_decorator=False):
        return [self.visit(stmt, allow_decorator=allow_decorator)
                for stmt in node.body]

    def visit_Name(self, node, loc):
        try:
            free, v = self.env.get_free(node.id)
            if free:
                if self.pull_free_variables:
                    v = self.new_variable(node.id)
                v = v.at(loc)
                self.free_variables[node.id] = v
            return v
        except NameError as e:
            # raise MyiaSyntaxError(loc, e.args[0])
            self.globals_accessed.add(node.id)
            return Symbol(node.id, namespace='global')

    def visit_Num(self, node, loc):
        return Literal(node.n)

    def visit_Return(self, node, loc):
        if self.return_error:
            raise MyiaSyntaxError(loc, self.return_error)
        self.returns = True
        return self.visit(node.value).at(loc)

    def visit_Slice(self, node, loc):
        return Apply(Symbol('slice'),
                     self.visit(node.lower) if node.lower else Literal(0),
                     self.visit(node.upper) if node.upper else Literal(None),
                     self.visit(node.step) if node.step else Literal(1))

    def visit_Str(self, node, loc):
        return Literal(node.s)

    def visit_Tuple(self, node, loc):
        return Tuple(self.visit(v) for v in node.elts).at(loc)

    def visit_Subscript(self, node, loc):
        return Apply(builtins.index, self.visit(node.value),
                     self.visit(node.slice.value),
                     location=loc)

    def visit_UnaryOp(self, node, loc):
        op = get_operator(node.op).at(loc)
        return Apply(op, self.visit(node.operand), location=loc)

    def visit_While(self, node, loc):
        fsym = self.global_env.gen.sym('#while')

        p = Parser(self, pull_free_variables=True)
        p.return_error = "While loops cannot contain return statements."
        test = p.visit(node.test)
        body = p.visit_body(node.body, True)
        in_vars = list(set(p.free_variables.keys()))
        out_vars = list(p.local_assignments)
        in_syms = [p.free_variables[v] for v in in_vars]

        initial_values = [p.env[v] for v in out_vars]
        new_body = If(test,
                      body(Apply(fsym, *[p.env[v] for v in in_vars])).at(loc),
                      Tuple(initial_values))

        if not self.dry:
            self.global_env[fsym.label] = Lambda(fsym.label, in_syms, new_body, location=loc)
        self.globals_accessed.add(fsym.label)

        tmp = self.gensym('#tmp').at(loc)
        val = Apply(fsym, *[self.env[v] for v in in_vars])
        stmts = [_Assign(tmp, val, None)]
        for i, v in enumerate(out_vars):
            stmt = self.make_assign(v, Apply(builtins.index, tmp, Literal(i)))
            stmts.append(stmt)
        return Begin(stmts)


def parse_function(fn):
    _, line = inspect.getsourcelines(fn)
    _, bindings = parse_source(inspect.getfile(fn),
                               line,
                               textwrap.dedent(inspect.getsource(fn)))
    return bindings

_global_envs = {}

def _get_global_env(url):
    return _global_envs.setdefault(url, Env(namespace='global'))

def parse_source(url, line, src):
    tree = ast.parse(src)
    p = Parser(Locator(url, line), _get_global_env(url), top_level=True)
    r = p.visit(tree, allow_decorator=True)
    # print(p.global_env.bindings)
    # print(p.globals_accessed)

    # for k, v in p.global_env.bindings.items():
    #     _validate(v)

    return r, p.global_env.bindings


def make_error_function(data):
    def _f(*args, **kwargs):
        raise Exception("Function {} is for internal use only.".format(data["name"]))
    _f.data = data
    return _f

def myia(fn):
    data = parse_function(fn)
    glob = fn.__globals__
    bindings = {k: make_error_function({"name": k, "ast": v, "globals": glob})
                for k, v in data.items()}
    glob.update(bindings)
    fn.data = bindings[fn.__name__].data
    fn.associates = bindings
    return fn


def _validate(node):
    if not node.location:
        print('Missing source location: {}'.format(node))
        if node.trace:
            t = node.trace[-1]
            print('  Definition at:')
            print('    ' + t.filename + ' line ' + str(t.lineno))
            print('    ' + t.line)
    for child in node.children():
        _validate(child)
