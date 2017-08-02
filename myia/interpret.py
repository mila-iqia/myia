from myia.ast import \
    MyiaASTNode, \
    Location, Symbol, Value, \
    Let, If, Lambda, Apply, Begin, Tuple, Closure
from .buche import HReprBase
from .symbols import builtins
import inspect


class BuiltinCollection:
    pass


myia_builtins = Symbol('myia_builtins', namespace='global')


global_env = {
    myia_builtins: BuiltinCollection()
}


###################
# Special objects #
###################


class PrimitiveImpl(HReprBase):
    def __init__(self, fn, name=None):
        self.argnames = inspect.getargs(fn.__code__).args
        self.nargs = len(self.argnames)
        self.fn = fn
        self.name = name or fn.__name__
        self.primal = None
        self.grad = None

    def __call__(self, *args):
        return self.fn(*args)

    def __str__(self):
        return f'Prim({self.name or self.fn})'

    def __repr__(self):
        return str(self)

    def __hrepr__(self, H, hrepr):
        return H.div['PrimitiveImpl'](
            H.div['class_title']('Primitive'),
            H.div['class_contents'](self.name or hrepr(self.fn))
        )


class FunctionImpl(HReprBase):
    def __init__(self, ast, envs):
        assert isinstance(ast, Lambda)
        self.argnames = [a.label for a in ast.args]
        self.nargs = len(ast.args)
        self.args = ast.args
        self.ast = ast
        self.instructions = VMCode(ast.body).instructions
        self.envs = envs
        self.primal = None
        self.grad = None
        node = ast

        def func(*args):
            assert(len(args) == len(node.args))
            return vm(self.instructions,
                      {s: arg for s, arg in zip(node.args, args)},
                      *self.envs)

        self._func = func

    def __call__(self, *args):
        return self._func(*args)

    def __str__(self):
        return f'Func({self.ast.ref or self.ast})'

    def __repr__(self):
        return str(self)

    def __hrepr__(self, H, hrepr):
        return H.div['FunctionImpl'](
            H.div['class_title']('Function'),
            H.div['class_contents'](hrepr(self.ast.ref or self.ast))
        )


class ClosureImpl(HReprBase):
    def __init__(self, fn, args):
        assert isinstance(fn, (PrimitiveImpl, FunctionImpl))
        self.argnames = [a for a in fn.argnames[len(args):]]
        self.nargs = fn.nargs - len(args)
        self.fn = fn
        self.args = args

    def __call__(self, *args):
        return self.fn(*self.args, *args)

    def __str__(self):
        return f'Clos({self.fn}, {self.args})'

    def __repr__(self):
        return str(self)

    def __hrepr__(self, H, hrepr):
        return H.div['ClosureImpl'](
            H.div['class_title']('Closure'),
            H.div['class_contents'](
                hrepr(self.fn),
                hrepr(self.args)
            )
        )


# class FunctionImpl:
#     def __init__(self, ast, bindings):
#         assert isinstance(ast, Lambda)
#         self.argnames = [a.label for a in ast.args]
#         self.nargs = len(ast.args)
#         self.ast = ast
#         self.bindings = bindings
#         self.primal = None
#         self.grad = None
#         node = ast

#         def func(*args):
#             assert(len(args) == len(node.args))
#             ev = Evaluator(
#                 {s: arg for s, arg in zip(node.args, args)},
#                 self.bindings
#             )
#             return ev.eval(node.body)

#         self._func = func

#     def __call__(self, *args):
#         return self._func(*args)

#     def __str__(self):
#         return f'Func({self.ast.ref or self.ast})'

#     def __repr__(self):
#         return str(self)


# class ClosureImpl:
#     def __init__(self, fn, args):
#         assert isinstance(fn, (PrimitiveImpl, FunctionImpl))
#         self.argnames = [a for a in fn.argnames[len(args):]]
#         self.nargs = fn.nargs - len(args)
#         self.fn = fn
#         self.args = args

#     def __call__(self, *args):
#         return self.fn(*self.args, *args)

#     def __str__(self):
#         return f'Clos({self.fn}, {self.args})'

#     def __repr__(self):
#         return str(self)


##########################
# Implementation helpers #
##########################


def impl(sym):
    def decorator(fn):
        prim = PrimitiveImpl(fn)
        global_env[sym] = prim
        setattr(global_env[myia_builtins],
                fn.__name__.lstrip('_'),
                prim)
        return prim
    return decorator


def myia_impl(sym):
    def decorator(orig_fn):
        r, bindings = parse_function0(orig_fn)
        fn = evaluate(r, bindings)
        global_env[sym] = fn
        setattr(global_env[myia_builtins],
                fn.__name__.lstrip('_'),
                fn)
        return fn
    return decorator


##############################################
# Implementations of myia's global functions #
##############################################


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


@impl(builtins.equal)
def equal(x, y):
    return x == y


@impl(builtins.less)
def less(x, y):
    return x < y


@impl(builtins.greater)
def greater(x, y):
    return x > y


@impl(builtins.len)
def _len(t):
    return len(t)


@impl(builtins.range)
def _range(t):
    return tuple(range(t))


@impl(builtins.index)
def index(t, i):
    return t[i]


@impl(builtins.first)
def first(t):
    return t[0]


@impl(builtins.second)
def second(t):
    return t[1]


@impl(builtins.getattr)
def _getattr(obj, attr):
    return getattr(obj, attr)


@impl(builtins.map)
def _map(f, xs):
    return tuple(map(f, xs))


@impl(builtins.enumerate)
def _enumerate(xs):
    return tuple(enumerate(xs))


@impl(builtins.switch)
def switch(cond, t, f):
    if cond:
        return t
    else:
        return f


@impl(builtins.identity)
def identity(x):
    return x


@impl(builtins.lazy_if)
def lazy_if(cond, t, f):
    if cond:
        return t()
    else:
        return f()


@impl(builtins.half_lazy_if)
def half_lazy_if(cond, t, f):
    # First branch is lazy, second is eager
    # Use case: when a branch is a constant
    if cond:
        return t()
    else:
        return f


##################################
# Class to Evaluate MyiaASTNodes #
##################################


def vm(instructions, *binding_groups):
    return VM(instructions, *binding_groups).result


class VMFrame:
    def __init__(self, instructions, envs):
        self.instructions = instructions
        self.pc = 0
        self.envs = ({},) + envs
        self.stack = [None]
        self.focus = None

    def top(self):
        return self.stack[-1]

    def take(self, n):
        if n == 0:
            return ()
        else:
            args = self.stack[-n:]
            del self.stack[-n:]
            return args

    def next_instruction(self):
        if self.pc >= len(self.instructions):
            return None
        self.pc += 1
        return self.instructions[self.pc - 1]

    def next(self):
        instr = self.next_instruction()
        if not instr:
            raise StopIteration()
        else:
            cmd, node, *args = instr
            self.focus = node
            method = getattr(self, 'instruction_' + cmd)
            return method(node, *args)

    def instruction_reduce(self, node, nargs):
        fn, *args = self.take(nargs + 1)
        if isinstance(fn, FunctionImpl):
            bind = {k: v for k, v in zip(fn.args, args)}
            return VMFrame(fn.instructions, (bind,) + fn.envs)
        elif isinstance(fn, ClosureImpl):
            self.stack.append(fn.fn)
            self.stack += fn.args
            self.stack += args
            return self.instruction_reduce(node, nargs + len(fn.args))
        else:
            value = fn(*args)
            self.stack.append(value)

    def instruction_tuple(self, node, nelems):
        self.stack.append(tuple(self.take(nelems)))

    def instruction_closure(self, node):
        args = self.stack.pop()
        fn = self.stack.pop()
        clos = ClosureImpl(fn, args)
        clos.primal = fn.primal
        self.stack.append(clos)

    def instruction_store(self, node, dest):
        value = self.stack.pop()
        self.envs[0][dest] = value

    def instruction_fetch(self, node):
        for env in self.envs:
            try:
                self.stack.append(env[node])
                return None
            except KeyError as err:
                pass
        raise KeyError(f'Could not resolve {node}')

    def instruction_push(self, node, value):
        self.stack.append(value)

    def instruction_lambda(self, node):
        self.stack.append(FunctionImpl(node, self.envs))


class VM:
    def __init__(self, instructions, *envs):
        # self.env = env
        self.frame = VMFrame(instructions, envs)
        self.frames = []
        self.result = self.eval()

    def eval(self):
        while True:
            try:
                new_frame = self.frame.next()
                if new_frame is not None:
                    self.frames.append(self.frame)
                    self.frame = new_frame
            except StopIteration:
                rval = self.frame.top()
                if not self.frames:
                    return rval
                else:
                    self.frame = self.frames.pop()
                    self.frame.stack.append(rval)
            except Exception as exc:
                self.error_unwind(exc)
                raise exc from None

    def error_unwind(self, exc):
        for i, frame in enumerate([self.frame] + self.frames):
            node = frame.focus
            if node:
                ann = {'error', f'error{i}'}
                node.annotations = node.annotations | ann


class VMCode:
    def __init__(self, node):
        self.node = node
        self.instructions = []
        self.process(self.node)

    def __call__(*args):
        self.frame = []
        for instruction, *params in self.instructions:
            method = getattr(self, 'instruction_' + instruction)
            method(node)
        return self.frame[-1]

    def instr(self, name, node, *params):
        self.instructions.append((name, node, *params))

    def process(self, node):
        cls = node.__class__.__name__
        method = getattr(self, 'process_' + cls)
        rval = method(node)

    def process_Apply(self, node):
        self.process(node.fn)
        for arg in node.args:
            self.process(arg)
        self.instr('reduce', node, len(node.args))

    def process_Begin(self, node):
        for stmt in node.stmts:
            self.process(stmt)

    def process_Closure(self, node):
        self.process(node.fn)
        for arg in node.args:
            self.process(arg)
        self.instr('tuple', node, len(node.args))
        self.instr('closure', node)

    def process_Lambda(self, node):
        self.instr('lambda', node)

    def process_Let(self, node):
        for k, v in node.bindings:
            self.process(v)
            self.instr('store', node, k)
        self.process(node.body)

    def process_Symbol(self, node):
        self.instr('fetch', node)

    def process_Tuple(self, node):
        for x in node.values:
            self.process(x)
        self.instr('tuple', node, len(node.values))

    def process_Value(self, node):
        self.instr('push', node, node.value)


# class Evaluator:
#     def __init__(self, env, global_env):
#         self.global_env = global_env
#         self.env = env

#     def eval(self, node):
#         cls = node.__class__.__name__
#         try:
#             method = getattr(self, 'eval_' + cls)
#         except AttributeError:
#             raise Exception(
#                 "Unrecognized node type for evaluation: {}".format(cls)
#             )
#         try:
#             rval = method(node)
#         except Exception as exc:
#             level = getattr(exc, 'level', 0)
#             exc.level = level + 1
#             node.annotations = node.annotations | {'error', f'error{level}'}
#             raise exc from None
#         return rval

#     def eval_Apply(self, node):
#         fn = self.eval(node.fn)
#         args = map(self.eval, node.args)
#         return fn(*args)

#     def eval_Begin(self, node):
#         rval = None
#         for stmt in node.stmts:
#             rval = self.eval(stmt)
#         return rval

#     def eval_Closure(self, node):
#         fn = self.eval(node.fn)
#         args = list(map(self.eval, node.args))
#         clos = ClosureImpl(fn, args)
#         clos.primal = fn.primal
#         return clos

#     def eval_If(self, node):
#         if self.eval(node.cond):
#             return self.eval(node.t)
#         else:
#             return self.eval(node.f)

#     def eval_Lambda(self, node):
#         return FunctionImpl(node, self.global_env)

#     def eval_Let(self, node):
#         for k, v in node.bindings:
#             self.env[k] = self.eval(v)
#         return self.eval(node.body)

#     def eval_Symbol(self, node):
#         try:
#             return self.env[node]
#         except KeyError:
#             return self.global_env[node]

#     def eval_Tuple(self, node):
#         return tuple(self.eval(x) for x in node.values)

#     def eval_Value(self, node):
#         return node.value


def evaluate(node, bindings):
    if isinstance(node, list):
        node, = node
    env = {**global_env}
    for k, v in bindings.items():
        if isinstance(v, MyiaASTNode):
            env[k] = vm(VMCode(v).instructions, env)
        else:
            env[k] = v
    return vm(VMCode(node).instructions, env)


# def old_evaluate(node, bindings):
#     if isinstance(node, list):
#         node, = node
#     env = {**global_env}
#     for k, v in bindings.items():
#         # env[Symbol(k, namespace='global')] = Evaluator({}, env).eval(v)
#         if isinstance(v, MyiaASTNode):
#             env[k] = Evaluator({}, env).eval(v)
#         else:
#             env[k] = v
#     return Evaluator({}, env).eval(node)
