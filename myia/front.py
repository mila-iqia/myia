
from myia.ast import \
    Location, Symbol, Literal, \
    LetRec, If, Lambda, Apply, Begin, Tuple
from myia.symbols import get_operator, builtins
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
        print(exception.location.traceback(), file=sys.stderr)
    else:
        _prevhook(exception_type, exception, traceback)
sys.excepthook = exception_handler


class Redirect:
    def __init__(self, key):
        self.key = key


class Env:
    def __init__(self, parent = None):
        self.parent = parent
        self.bindings = {}

    def fork(self):
        return Env(self)

    def get(self, name, redirect=True):
        if name in self.bindings:
            result = self.bindings[name]
        elif self.parent is None:
            raise Exception("Undeclared variable: {}".format(name))
        else:
            result = self.parent[name]
        if redirect and isinstance(result, Redirect):
            return self[result.key]
        else:
            return result

    def set(self, bindings):
        self.bindings.update(bindings)

    def __getitem__(self, name):
        return self.get(name)

    def __setitem__(self, name, value):
        self.bindings[name] = value


class Locator:
    def __init__(self, url, line_offset):
        self.url = url
        self.line_offset = line_offset

    def __call__(self, node):
        return Location(self.url, node.lineno + self.line_offset - 1, node.col_offset)


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
        return method(node, loc, **kwargs)


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

    def __init__(self, parent):
        self.free_variables = []
        self.local_assignments = set()
        self.returns = False
        
        if isinstance(parent, Locator):
            self.parent = None
            self.env = Env()
            self.varcount = {}
            super().__init__(parent)
        else:
            self.parent = parent
            self.env = parent.env.fork()
            self.varcount = parent.varcount
            super().__init__(parent.locator)

    def gensym(self, name):
        if name in self.varcount:
            self.varcount[name] += 1
            return '{}#{}'.format(name, self.varcount[name])
        else:
            self.varcount[name] = 0
            return name

    def visit_arguments(self, args):
        return [self.visit(arg) for arg in args.args]

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

    def visit_Return(self, node, loc):
        self.returns = True
        return self.visit(node.value).at(loc)

    def new_variable(self, base_name):
        sym = Symbol(self.gensym(base_name))
        self.env.set({base_name: Redirect(sym.label)})
        # The following statement can override the previous, if sym.label == base_name
        # That is fine and intended.
        self.env.set({sym.label: sym})
        return sym

    def make_assign(self, base_name, value, location=None):
        sym = self.new_variable(base_name)
        self.local_assignments.add(base_name)
        return _Assign(sym, value, location)

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
                tmp = Symbol(self.gensym('#tmp'))
                stmts = [_Assign(tmp, val, None)]
                for i, a in enumerate(ass):
                    stmt = self.make_assign(a, Apply(builtins.index, tmp, Literal(i)))
                    stmts.append(stmt)
                return Begin(stmts)

    def visit_Assign(self, node, loc):
        targ, = node.targets
        if isinstance(targ, ast.Tuple):
            raise MyiaSyntaxError(loc, "Deconstructing assignment is not supported.")
        val = self.visit(node.value)
        return self.make_assign(targ.id, val, loc)

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
        subp = Parser(self)
        args = [Symbol(self.gensym(arg.arg)) for arg in node.args.args]
        subp.env.set({arg.arg: s for arg, s in zip(node.args.args, args)})
        result = subp.visit_body(node.body)
        if subp.free_variables:
            v, _ = items(subp.free_variables)[0]
            raise MyiaSyntaxError(v.location, "Functions cannot have free variables.")
        if not subp.returns:
            raise MyiaSyntaxError(loc, "Function does not return a value.")
        return Lambda(node.name,
                      args,
                      result,
                      location=loc)

    def visit_Lambda(self, node, loc):
        return Lambda("lambda",
                      self.visit_arguments(node.args),
                      self.visit(node.body),
                      location=loc)

    def visit_Expr(self, node, loc):
        return self.visit(node.value)

    def visit_Name(self, node, loc):
        try:
            return self.env[node.id]
        except Exception as e:
            raise MyiaSyntaxError(loc, e.args[0])

    def visit_Num(self, node, loc):
        return Literal(node.n)

    def visit_Str(self, node, loc):
        return Literal(node.s)

    def visit_IfExp(self, node, loc):
        return If(self.visit(node.test),
                  self.visit(node.body),
                  self.visit(node.orelse),
                  location=loc)

    def visit_BinOp(self, node, loc):
        op = get_operator(node.op)
        return Apply(op, self.visit(node.left), self.visit(node.right), location=loc)

    def visit_BoolOp(self, node, loc):
        left, right = node.values
        if isinstance(node.op, ast.And):
            return If(self.visit(left), self.visit(right), Literal(False))
        elif isinstance(node.op, ast.Or):
            return If(self.visit(left), Literal(True), self.visit(right))
        else:
            raise MyiaSyntaxError(loc, "Unknown operator: {}".format(node.op))

    def visit_Compare(self, node, loc):
        ops = [get_operator(op) for op in node.ops]
        if len(ops) == 1:
            return Apply(ops[0], self.visit(node.left), self.visit(node.comparators[0]))
        else:
            raise MyiaSyntaxError(loc,
                                  "Comparisons must have a maximum of two operands")
        
    def visit_Subscript(self, node, loc):
        return Apply(builtins.index, self.visit(node.value),
                     self.visit(node.slice.value),
                     location=loc)

    def visit_Call(self, node, loc):
        if (len(node.keywords) > 0):
            raise MyiaSyntaxError(loc, "Keyword arguments are not allowed.")
        return Apply(self.visit(node.func),
                     *[self.visit(arg) for arg in node.args],
                     location=loc)
        

    def visit_ListComp(self, node, loc):
        if len(node.generators) > 1:
            raise MyiaSyntaxError(loc,
                "List comprehensions can only iterate over a single target")
        gen = node.generators[0]
        if len(gen.ifs) > 0:
            test1, *others = reversed(gen.ifs)
            cond = self.visit(test1)
            for test in others:
                cond = If(self.visit(test), cond, Literal(False))
            arg = Apply(builtins.filter,
                        Lambda("filtercmp", [self.visit(gen.target)], cond),
                        self.visit(gen.iter))
        else:
            arg = self.visit(gen.iter)
        return Apply(builtins.map,
                     Lambda("listcmp", [self.visit(gen.target)], self.visit(node.elt)),
                     arg,
                     location=loc)

    def visit_arg(self, node, loc):
        return Symbol(node.arg, location=loc)


def parse_function(fn):
    _, line = inspect.getsourcelines(fn)
    return parse_source(inspect.getfile(fn), line, textwrap.dedent(inspect.getsource(fn)))


def parse_source(url, line, src):
    tree = ast.parse(src).body[0]
    return Parser(Locator(url, line)).visit(tree, allow_decorator=True)


def myia(fn):
    return parse_function(fn)
