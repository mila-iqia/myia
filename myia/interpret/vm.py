from typing import Dict, Callable, List, Any, Union, Tuple as TupType, Optional

from types import FunctionType
from ..stx import \
    MyiaASTNode, ParseEnv, \
    Location, Symbol, ValueNode, LambdaNode, maptup2, \
    add_source
from ..lib import Closure as ClosureImpl, Primitive as PrimitiveImpl, \
    IdempotentMappable
from ..symbols import builtins, object_map, update_object_map
from ..util import EventDispatcher, BucheDb, HReprBase, buche
from functools import reduce
from ..impl.main import impl_bank, GlobalEnv
from ..parse import parse_function
from ..lib import Pending


# When a LambdaNode is made into a FunctionImpl, we will
# cache it. This mostly avoids recomputing gradients,
# since FunctionImpl is the structure that stores
# pointers to them.
compile_cache: Dict[LambdaNode, 'FunctionImpl'] = {}
EnvT = Dict[Symbol, Any]
root_globals = impl_bank['interp']
add_source(':builtin', root_globals)


vm_genv = GlobalEnv(impl_bank['interp'])


_loaded = False


def load():
    update_object_map()
    global _loaded
    if not _loaded:
        _loaded = True
        # The following two imports fill impl_bank['interp']
        # as a side-effect.
        from ..impl.impl_interp import _
        from ..impl.impl_bprop import _
        # root_globals.update(impl_bank['interp'])
        global root_globals
        root_globals = impl_bank['interp']


###################
# Special objects #
###################


def translate_node(v):
    if isinstance(v, MyiaASTNode):
        return v, {}

    try:
        return object_map[v], {}
    except:
        pass

    if isinstance(v, (int, float, FunctionImpl, PrimitiveImpl)):
        return ValueNode(v), {}
    elif isinstance(v, (type, FunctionType)):
        try:
            sym, genv = parse_function(v)
        except (TypeError, OSError):
            raise ValueError(f'Myia cannot translate function: {v}')
        return genv[sym], genv

    raise ValueError(f'Myia cannot process value: {v}')


def process_value(value, vm):
    node, bindings = translate_node(value)
    if isinstance(node, LambdaNode):
        cv = vm.compile_cache.get(node, None)
        if cv is None:
            cv = vm.evaluate(node)
            assert isinstance(cv, FunctionImpl)
            vm.compile_cache[node] = cv
        return cv

    elif isinstance(node, ValueNode):
        return node.value

    elif isinstance(node, Symbol):
        return root_globals[node]

    else:
        raise ValueError(f'Myia cannot process: {self.value}')


class FunctionImpl(HReprBase, IdempotentMappable):
    """
    Represents a Myia-transformed function.
    """
    def __init__(self, ast: LambdaNode, envs: List[EnvT]) -> None:
        assert isinstance(ast, LambdaNode)
        assert isinstance(envs, list)
        self.argnames = [a.label for a in ast.args]
        self.nargs = len(ast.args)
        self.ast = ast
        self.code = VMCode(ast.body)
        self.envs = envs
        self.primal_sym = ast.primal
        self.grad: Callable[[int], FunctionImpl] = None

    def debug(self, args, debugger):
        ast = self.ast
        assert len(args) == len(ast.args)
        return run_vm(self.code,
                      {s: arg for s, arg in zip(ast.args, args)},
                      *self.envs,
                      debugger=debugger)

    def __call__(self, *args):
        return self.debug(args, None)

    def __str__(self):
        return f'Func({self.ast.ref or self.ast})'

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.ast)

    def __eq__(self, other):
        return type(other) is FunctionImpl \
            and self.ast == other.ast

    def __add__(self, other):
        # TODO: Fix the following issue, which happens sometimes.
        #       I believe the main reason is that for the backpropagator,
        #       Grad builds a closure on the wrong function (?)
        # if self != other:
        #     raise Exception('The functions being added are different.')
        return self

    def __hrepr__(self, H, hrepr):
        return H.div['FunctionImpl'](
            H.div['class_title']('Function'),
            H.div['class_contents'](hrepr(self.ast.ref or self.ast))
        )


##################################
# Class to Evaluate MyiaASTNodes #
##################################


class Instruction:
    """
    An instruction for the stack-based VM.

    Attributes:
        command: The instruction name.
        node: The Myia node that this instruction is computing.
        args: Instruction-specific arguments.
    """
    def __init__(self,
                 command: str,
                 node: MyiaASTNode,
                 *args: Any) -> None:
        self.command = command
        self.node = node
        self.args = args

    def __str__(self):
        args = ", ".join(map(str, self.args))
        return f'{self.command}:{args}'


class VMCode(HReprBase):
    """
    Compile a MyiaASTNode into a list of instructions compatible
    with the stack-based VM.

    See VMFrame's ``instruction_<name>`` methods for more
    information.

    Attributes:
        node: The original node.
        instructions: A list of instructions to implement this
            node's behavior.
    """
    def __init__(self,
                 node: MyiaASTNode,
                 instructions: List[Instruction] = None) -> None:
        self.node = node
        if instructions is None:
            self.instructions: List[Instruction] = []
            self.process(self.node)
        else:
            self.instructions = instructions

    def instr(self, name, node, *args) -> None:
        self.instructions.append(Instruction(name, node, *args))

    def process(self, node) -> None:
        # Dispatch to process_<node_type>
        cls = node.__class__.__name__
        method = getattr(self, 'process_' + cls)
        rval = method(node)

    def process_ApplyNode(self, node) -> None:
        self.process(node.fn)
        for arg in node.args:
            self.process(arg)
        self.instr('reduce', node, len(node.args))

    def process_BeginNode(self, node) -> None:
        for stmt in node.stmts:
            self.process(stmt)

    def process_ClosureNode(self, node) -> None:
        self.process(node.fn)
        self.process(builtins.mktuple)
        for arg in node.args:
            self.process(arg)
        # self.instr('tuple', node, len(node.args))
        self.instr('reduce', node, len(node.args))
        self.instr('closure', node)

    def process_LambdaNode(self, node) -> None:
        self.instr('lambda', node)

    def process_LetNode(self, node) -> None:
        for k, v in node.bindings:
            self.process(v)
            self.instr('store', node, k)
        self.process(node.body)

    def process_Symbol(self, node) -> None:
        self.instr('fetch', node, node)

    def process_TupleNode(self, node) -> None:
        # for x in node.values:
        #     self.process(x)
        # self.instr('tuple', node, len(node.values))
        self.process(builtins.mktuple)
        for arg in node.values:
            self.process(arg)
        self.instr('reduce', node, len(node.values))

    def process_ValueNode(self, node) -> None:
        self.instr('push', node, node.value)

    def __hrepr__(self, H, hrepr):
        rows = []
        for instr in self.instructions:
            row = H.tr()
            for x in (instr.command, *instr.args):
                row = row(H.td(hrepr(x)))
            rows.append(row)
        return H.table['VMCodeInstructions'](*rows)


class VM(EventDispatcher):
    """
    Stack-based virtual machine. Evaluates the given code
    in the constructor and puts the value in the ``result``
    attribute.

    Arguments:
        code: The compiled VMCode to run.
        envs: A list of environments that can be used to resolve
            the value of a symbol.
        emit_events: Whether to emit events on each instruction
            run or not.

    Attributes:
        result: The result of the evaluation.
    """
    def __init__(self,
                 code: VMCode,
                 *envs: EnvT,
                 debugger: BucheDb = None,
                 emit_events=True) -> None:
        super().__init__(self, emit_events)
        self.compile_cache = compile_cache
        self.debugger = debugger
        self.do_emit_events = emit_events
        # Current frame
        self.frame = VMFrame(self, code, list(envs))
        # Stack of previous frames (excludes current one)
        self.frames: List[VMFrame] = []
        if self.do_emit_events:
            self.emit_new_frame(self.frame)
        self.result = self.eval()

    def evaluate(self, v):
        return evaluate(v)

    def eval(self) -> Any:
        while True:
            try:
                # VMFrame does most of the work.
                new_frame = self.frame.next()
                if new_frame is not None:
                    # When the current frame gives us a new frame,
                    # we push the old one on the stack and start
                    # processing the new one.
                    if not self.frame.done():
                        # We push the current frame only if it's
                        # not done (this implement tail calls).
                        self.frames.append(self.frame)
                    self.frame = new_frame
                    if self.do_emit_events:
                        self.emit_new_frame(self.frame)
            except StopIteration:
                # The result of a frame's evaluation is the value at
                # the top of its stack.
                rval = self.frame.top()
                if not self.frames:
                    # We are done!
                    return rval
                else:
                    # We push the result on the previous frame's stack
                    # and we resume execution.
                    self.frame = self.frames.pop()
                    self.frame.stack.append(rval)
            except Exception as exc:
                if self.do_emit_events:
                    self.emit_error(exc)
                raise exc from None


class VMFrame(HReprBase):
    """
    Computation frame. There is one frame for each FunctionImpl
    called. A frame has its own stack, while the VM operates on
    a stack of VMFrames.

    Compute a frame's next instruction with ``next()``,
    which may return a new VMFrame to the VM to compute something it
    needs, or throw StopIteration if it is done, in which case its
    return value is at the top of its stack.
    """
    def __init__(self,
                 vm: VM,
                 code: VMCode,
                 envs: List[EnvT]) -> None:
        envs = list(envs)  # TODO: remove
        self.vm = vm
        self.code = code
        self.instructions = code.instructions
        # Program counter: index of the next instruction to execute.
        self.pc = 0
        # Environment to store local bindings.
        self.storage_env: EnvT = {}
        self.envs: List[EnvT] = [self.storage_env] + envs
        self.stack: List[Any] = [None]
        # Node being executed, mostly for debugging.
        self.focus: MyiaASTNode = None
        self.signature: Any = None

    def done(self) -> bool:
        """
        Whether all instructions have been executed or not.
        """
        return self.pc >= len(self.instructions)

    def top(self) -> Any:
        """
        Value at the top of the stack.
        """
        return self.stack[-1]

    def take(self, n: int) -> List[Any]:
        """
        Pop n values from the stack and return them.
        """
        if n == 0:
            return []
        else:
            args = self.stack[-n:]
            del self.stack[-n:]
            return args

    def push(self, *values):
        self.stack += list(values)

    def pop(self):
        return self.stack.pop()

    def next_instruction(self) -> Optional[Instruction]:
        """
        Get the next instruction and advance the program
        counter.

        Returns:
           * None if we are done.
           * The next instruction otherwise.
        """
        if self.pc >= len(self.instructions):
            return None
        self.pc += 1
        return self.instructions[self.pc - 1]

    def next(self) -> Optional['VMFrame']:
        """
        Execute the next instruction.

        Returns:
            * ``None`` for most operations.
            * A ``VMFrame``. The VM should execute that frame
              and push its result to this frame before resuming
              execution.
        """
        instr = self.next_instruction()
        if not instr:
            raise StopIteration()
        else:
            self.focus = instr.node
            mname = 'instruction_' + instr.command
            if self.vm.debugger:
                if 'break' in self.focus.annotations:
                    self.vm.debugger.buche(self)
                    self.vm.debugger.set_trace()
            if self.vm.do_emit_events:
                self.vm.emit(mname, self, instr.node, *instr.args)
                self.vm.emit_instruction(self, instr)
            method = getattr(self, mname)
            return method(instr.node, *instr.args)

    def instruction_reduce(self, node, nargs) -> Optional['VMFrame']:
        """
        * Pop ``nargs`` values from the stack, call them ``args``
        * Pop the next value, call it ``fn``
        * If ``fn`` is a ``FunctionImpl``, we can run it with the VM.
          Make a new VMFrame for it and return it. This is important
          because we don't want to grow the Python stack.
        * Otherwise, it's a primitive. Call ``fn(*args)`` and push
          the result.
        """
        fn, *args = self.take(nargs + 1)
        if isinstance(fn, FunctionImpl):
            bind: EnvT = {k: v for k, v in zip(fn.ast.args, args)}
            return self.__class__(self.vm, fn.code, [bind] + fn.envs)
        elif isinstance(fn, ClosureImpl):
            self.push(fn.fn, *fn.args, *args)
            return self.instruction_reduce(node, nargs + len(fn.args))
        else:
            value = fn(*args)
            self.push(value)
            return None

    # def instruction_tuple(self, node, nelems) -> None:
    #     """
    #     Pop ``nelems`` values from the stack and push a
    #     tuple of these values.
    #     """
    #     self.push(tuple(self.take(nelems)))

    def instruction_closure(self, node) -> None:
        """
        Pop a tuple of arguments, and a function, and push
        ``ClosureImpl(fn, args)``
        """
        fn, args = self.take(2)
        clos = ClosureImpl(fn, args)
        self.stack.append(clos)

    def instruction_store(self, node, dest) -> None:
        """
        Pop a value and store it in the local environment
        under ``dest``. ``dest`` may be a Symbol or a tree
        of Symbols, represented as a Tuple.
        """
        value = self.pop()

        def store(dest, val):
            if isinstance(dest, Symbol):
                self.envs[0][dest] = val
            else:
                raise TypeError(f'Cannot store into {dest}.')

        maptup2(store, dest, value)

    def instruction_fetch(self, node, sym) -> None:
        """
        Get the value for symbol ``sym`` from one of the
        environments, starting with the local environment,
        and push it on the stack.
        """
        for env in self.envs:
            try:
                v = env[sym]
                if isinstance(v, LambdaNode):
                    v = Pending(v)
                if isinstance(v, Pending):
                    v = process_value(v.value, self.vm)
                    env[sym] = v
                self.push(v)
                return None
            except KeyError as err:
                pass
        raise KeyError(f'Could not resolve {sym} ({sym.namespace})')

    def instruction_push(self, node, value) -> None:
        """
        Push ``value`` on the stack.
        """
        self.push(value)

    def instruction_lambda(self, node) -> None:
        """
        Create a FunctionImpl from the given node and push
        it on the stack.
        """
        fimpl = FunctionImpl(node, self.envs)
        self.push(fimpl)

    def __hrepr__(self, H, hrepr):
        views = H.tabbedView['hrepr-VMFrame']()
        env = {}
        for e in reversed(self.envs):
            env.update(e)
        eviews = H.tabbedView()
        for k, v in env.items():
            view = H.view(H.tab(hrepr(k)), H.pane(hrepr(v)))
            eviews = eviews(view)
        views = views(H.view(H.tab('Focus'), H.pane(hrepr(self.focus))))
        views = views(H.view(H.tab('Node'), H.pane(hrepr(self.code.node))))
        views = views(H.view(H.tab('Code'), H.pane(hrepr(self.code))))
        views = views(H.view(H.tab('Stack'), H.pane(hrepr(self.stack))))
        views = views(H.view(H.tab('Env'), H.pane(eviews)))
        return views


def run_vm(code: VMCode,
           *binding_groups: EnvT,
           debugger: BucheDb = None) -> Any:
    """
    Execute the VM on the given code.
    """
    return VM(code, *binding_groups, debugger=debugger).result


def evaluate(node: MyiaASTNode,
             parse_env: ParseEnv = None,
             debugger: BucheDb = None) -> Any:
    """
    Evaluate the given MyiaASTNode in the given ``parse_env``.
    If ``parse_env`` is None, it will be extracted from the node
    itself (Parser stores a LambdaNode's global environment in its
    ``global_env`` field).
    """
    load()
    if isinstance(node, LambdaNode):
        parse_env = node.global_env
    assert parse_env is not None
    # envs = (parse_env.bindings, root_globals)
    envs = (vm_genv,)
    return run_vm(VMCode(node), *envs, debugger=debugger)


def evaluate2(node: MyiaASTNode,
              env: Dict[Symbol, Any],
              debugger: BucheDb = None) -> Any:
    load()
    # envs = (env, root_globals)
    envs = (vm_genv,)
    return run_vm(VMCode(node), *envs, debugger=debugger)
