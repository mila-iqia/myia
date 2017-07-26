from myia.ast import \
    MyiaASTNode, \
    Location, Symbol, Value, \
    Let, If, Lambda, Apply, Begin, Tuple, Closure
from .symbols import builtins
import inspect

global_env = {}


###################
# Special objects #
###################


class PrimitiveImpl:
    def __init__(self, fn):
        self.argnames = inspect.getargs(fn.__code__).args
        self.nargs = len(self.argnames)
        self.fn = fn
        self.primal = None
        self.grad = None

    def __call__(self, *args):
        return self.fn(*args)


class FunctionImpl:
    def __init__(self, ast, bindings):
        assert isinstance(ast, Lambda)
        self.argnames = [a.label for a in ast.args]
        self.nargs = len(ast.args)
        self.ast = ast
        self.bindings = bindings
        self.primal = None
        self.grad = None
        node = ast

        def func(*args):
            ev = Evaluator(
                {s: arg for s, arg in zip(node.args, args)},
                self.bindings
            )
            return ev.eval(node.body)

        self._func = func

    def __call__(self, *args):
        return self._func(*args)


class ClosureImpl:
    def __init__(self, fn, args):
        assert isinstance(fn, (PrimitiveImpl, FunctionImpl))
        self.argnames = [a for a in fn.argnames[len(args):]]
        self.nargs = fn.nargs - len(args)
        self.fn = fn
        self.args = args

    def __call__(self, *args):
        return self.fn(*self.args, *args)


############################################
# Implementations of arithmetic primitives #
############################################


def impl(sym):
    def decorator(fn):
        prim = PrimitiveImpl(fn)
        global_env[sym] = prim
        return prim
    return decorator


@impl(builtins.add)
def add(x, y):
    return x + y


@impl(builtins.subtract)
def subtract(x, y):
    return x - y


@impl(builtins.multiply)
def multiply(x, y):
    return x * y


@impl(builtins.divide)
def divide(x, y):
    return x / y


@impl(builtins.unary_subtract)
def unary_subtract(x):
    return -x


@impl(builtins.less)
def less(x, y):
    return x < y


@impl(builtins.greater)
def greater(x, y):
    return x > y


@impl(builtins.index)
def index(t, i):
    return t[i]


@impl(builtins.map)
def _map(f, xs):
    return list(map(f, xs))


class Evaluator:
    def __init__(self, env, global_env):
        self.global_env = global_env
        self.env = env

    def eval(self, node):
        cls = node.__class__.__name__
        try:
            method = getattr(self, 'eval_' + cls)
        except AttributeError:
            raise Exception(
                "Unrecognized node type for evaluation: {}".format(cls)
            )
        rval = method(node)
        return rval

    def eval_Apply(self, node):
        fn = self.eval(node.fn)
        args = map(self.eval, node.args)
        return fn(*args)

    def eval_Begin(self, node):
        rval = None
        for stmt in node.stmts:
            rval = self.eval(stmt)
        return rval

    def eval_Closure(self, node):
        fn = self.eval(node.fn)
        args = list(map(self.eval, node.args))
        clos = ClosureImpl(fn, args)
        clos.primal = fn.primal
        return clos

    def eval_If(self, node):
        if self.eval(node.cond):
            return self.eval(node.t)
        else:
            return self.eval(node.f)

    def eval_Lambda(self, node):
        return FunctionImpl(node, self.global_env)

    def eval_Let(self, node):
        for k, v in node.bindings:
            self.env[k] = self.eval(v)
        return self.eval(node.body)

    def eval_Symbol(self, node):
        try:
            return self.env[node]
        except KeyError:
            return self.global_env[node]

    def eval_Tuple(self, node):
        return tuple(self.eval(x) for x in node.values)

    def eval_Value(self, node):
        return node.value


def evaluate(node, bindings):
    if isinstance(node, list):
        node, = node
    env = {**global_env}
    for k, v in bindings.items():
        # env[Symbol(k, namespace='global')] = Evaluator({}, env).eval(v)
        if isinstance(v, MyiaASTNode):
            env[k] = Evaluator({}, env).eval(v)
        else:
            env[k] = v
    return Evaluator({}, env).eval(node)
