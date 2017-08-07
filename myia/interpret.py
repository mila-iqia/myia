from typing import Dict, Callable, List, Any, Union, Tuple as TupType, Optional

from myia.ast import \
    MyiaASTNode, \
    Location, Symbol, Value, \
    Let, If, Lambda, Apply, Begin, Tuple, Closure
from .front import parse_function, ParseEnv
from .buche import HReprBase
from .symbols import builtins
from .event import EventDispatcher
from functools import reduce
import inspect


class BuiltinCollection:
    pass


myia_builtins = builtins.myia_builtins


root_globals: Dict[Any, Any] = {
    myia_builtins: BuiltinCollection()
}


compile_cache: Dict[Lambda, 'FunctionImpl'] = {}


EnvT = Dict[Union[str, Symbol], Any]


###################
# Special objects #
###################


class PrimitiveImpl(HReprBase):
    def __init__(self, fn: Callable, name=None) -> None:
        argn = inspect.getargs(fn.__code__).args  # type: ignore
        self.argnames: List[str] = argn
        self.nargs = len(self.argnames)
        self.fn = fn
        self.name = name or fn.__name__
        self.primal_sym = None
        self.grad: Callable[[int], FunctionImpl] = None

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
    def __init__(self, ast: Lambda, envs: List[EnvT]) -> None:
        assert isinstance(ast, Lambda)
        self.argnames = [a.label for a in ast.args]
        self.nargs = len(ast.args)
        self.args = ast.args
        self.ast = ast
        self.code = VMCode(ast.body)
        self.envs = envs
        self.primal_sym = ast.primal
        self.grad: Callable[[int], FunctionImpl] = None
        node = ast

        def func(*args):
            assert(len(args) == len(node.args))
            return vm(self.code,
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
    def __init__(self,
                 fn: Union[PrimitiveImpl, FunctionImpl],
                 args: List[Any]) -> None:
        if not isinstance(fn, (PrimitiveImpl, FunctionImpl)):
            raise TypeError(f'Wrong fn for ClosureImpl: {fn}')
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


##########################
# Implementation helpers #
##########################


def impl(sym):
    def decorator(fn):
        prim = PrimitiveImpl(fn)
        root_globals[sym] = prim
        setattr(root_globals[myia_builtins],
                fn.__name__.lstrip('_'),
                prim)
        return prim
    return decorator


def myia_impl(sym):
    def decorator(orig_fn):
        r, genv = parse_function(orig_fn)
        fn = evaluate(r, genv)
        root_globals[sym] = fn
        setattr(root_globals[myia_builtins],
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


@impl(builtins.reduce)
def _reduce(f, xs):
    return reduce(f, xs)


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


##################################
# Class to Evaluate MyiaASTNodes #
##################################


class Instruction:
    def __init__(self,
                 command: str,
                 node: MyiaASTNode,
                 *args: Any) -> None:
        self.command = command
        self.node = node
        self.args = args


class VMCode(HReprBase):
    def __init__(self, node: MyiaASTNode) -> None:
        self.node = node
        self.instructions: List[Instruction] = []
        self.process(self.node)

    def instr(self, name, node, *args) -> None:
        self.instructions.append(Instruction(name, node, *args))

    def process(self, node) -> None:
        cls = node.__class__.__name__
        method = getattr(self, 'process_' + cls)
        rval = method(node)

    def process_Apply(self, node) -> None:
        self.process(node.fn)
        for arg in node.args:
            self.process(arg)
        self.instr('reduce', node, len(node.args))

    def process_Begin(self, node) -> None:
        for stmt in node.stmts:
            self.process(stmt)

    def process_Closure(self, node) -> None:
        self.process(node.fn)
        for arg in node.args:
            self.process(arg)
        self.instr('tuple', node, len(node.args))
        self.instr('closure', node)

    def process_Lambda(self, node) -> None:
        self.instr('lambda', node)

    def process_Let(self, node) -> None:
        for k, v in node.bindings:
            self.process(v)
            self.instr('store', node, k)
        self.process(node.body)

    def process_Symbol(self, node) -> None:
        self.instr('fetch', node, node)

    def process_Tuple(self, node) -> None:
        for x in node.values:
            self.process(x)
        self.instr('tuple', node, len(node.values))

    def process_Value(self, node) -> None:
        self.instr('push', node, node.value)

    def __hrepr__(self, H, hrepr):
        rows = []
        for cmd, _, *args in self.instructions:
            row = H.tr()
            for x in (cmd, *args):
                row = row(H.td(hrepr(x)))
            rows.append(row)
        return H.table['VMCodeInstructions'](*rows)


class VM(EventDispatcher):
    def __init__(self,
                 code: VMCode,
                 *envs: EnvT,
                 emit_events=True) -> None:
        super().__init__(self, emit_events)
        self.do_emit_events = emit_events
        self.frame = VMFrame(self, code, list(envs))
        self.frames: List[VMFrame] = []
        if self.do_emit_events:
            self.emit_new_frame(self.frame)
        self.result = self.eval()

    def eval(self) -> Any:
        while True:
            try:
                new_frame = self.frame.next()
                if new_frame is not None:
                    self.frames.append(self.frame)
                    self.frame = new_frame
                    if self.do_emit_events:
                        self.emit_new_frame(self.frame)
            except StopIteration:
                rval = self.frame.top()
                if not self.frames:
                    return rval
                else:
                    self.frame = self.frames.pop()
                    self.frame.stack.append(rval)
            except Exception as exc:
                if self.do_emit_events:
                    self.emit_error(exc)
                raise exc from None


class VMFrame(HReprBase):
    def __init__(self,
                 vm: VM,
                 code: VMCode,
                 envs: List[EnvT]) -> None:
        envs = list(envs)  # TODO: remove
        self.vm = vm
        self.code = code
        self.instructions = code.instructions
        self.pc = 0
        self.storage_env: EnvT = {}
        self.envs: List[EnvT] = [self.storage_env] + envs
        self.stack: List[Any] = [None]
        self.focus: MyiaASTNode = None

    def top(self) -> Any:
        return self.stack[-1]

    def take(self, n: int) -> List[Any]:
        if n == 0:
            return []
        else:
            args = self.stack[-n:]
            del self.stack[-n:]
            return args

    def next_instruction(self) -> Optional[Instruction]:
        if self.pc >= len(self.instructions):
            return None
        self.pc += 1
        return self.instructions[self.pc - 1]

    def next(self) -> Optional['VMFrame']:
        instr = self.next_instruction()
        if not instr:
            raise StopIteration()
        else:
            self.focus = instr.node
            mname = 'instruction_' + instr.command
            if self.vm.do_emit_events:
                self.vm.emit(mname, self, instr.node, *instr.args)
                self.vm.emit_instruction(self, instr)
            method = getattr(self, mname)
            return method(instr.node, *instr.args)

    def instruction_reduce(self, node, nargs) -> Optional['VMFrame']:
        fn, *args = self.take(nargs + 1)
        if isinstance(fn, FunctionImpl):
            bind: EnvT = {k: v for k, v in zip(fn.args, args)}
            return VMFrame(self.vm, fn.code, [bind] + fn.envs)
        elif isinstance(fn, ClosureImpl):
            self.stack.append(fn.fn)
            self.stack += fn.args
            self.stack += args
            return self.instruction_reduce(node, nargs + len(fn.args))
        else:
            value = fn(*args)
            self.stack.append(value)
            return None

    def instruction_tuple(self, node, nelems) -> None:
        self.stack.append(tuple(self.take(nelems)))

    def instruction_closure(self, node) -> None:
        args = self.stack.pop()
        fn = self.stack.pop()
        clos = ClosureImpl(fn, args)
        self.stack.append(clos)

    def instruction_store(self, node, dest) -> None:
        value = self.stack.pop()
        self.envs[0][dest] = value

    def instruction_fetch(self, node, sym) -> None:
        for env in self.envs:
            try:
                v = env[sym]
                if isinstance(v, Lambda):
                    cv = compile_cache.get(v, None)
                    if cv is None:
                        cv = evaluate(v)
                        assert isinstance(cv, FunctionImpl)
                        compile_cache[v] = cv
                    v = cv
                self.stack.append(v)
                return None
            except KeyError as err:
                pass
        raise KeyError(f'Could not resolve {sym}')

    def instruction_push(self, node, value) -> None:
        self.stack.append(value)

    def instruction_lambda(self, node) -> None:
        fimpl = FunctionImpl(node, self.envs)
        if node.primal:
            fimpl.primal_sym = node.primal
        self.stack.append(fimpl)

    def __hrepr__(self, H, hrepr):
        ref = getattr(self.code.node, 'ref', self.code.node)
        return H.div['VMFrame'](
            H.div['class_title']('VMFrame'),
            H.div['class_contents'](hrepr(ref))
        )


def vm(code: VMCode, *binding_groups: EnvT) -> Any:
    return VM(code, *binding_groups).result


def evaluate(node: MyiaASTNode, parse_env: ParseEnv = None) -> Any:
    if isinstance(node, Lambda):
        parse_env = node.global_env
    assert parse_env is not None
    envs = (parse_env.bindings, root_globals)
    return vm(VMCode(node), *envs)
