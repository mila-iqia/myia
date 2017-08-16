from typing import Dict, Callable, List, Any, Union, Tuple as TupType, Optional

from ..stx import \
    MyiaASTNode, \
    Location, Symbol, Value, \
    Let, Lambda, Apply, Begin, Tuple, Closure, maptup2
from ..front import parse_function, ParseEnv
from ..symbols import builtins
from ..util import EventDispatcher, BucheDb, HReprBase, buche
from .runtime import FunctionImpl, ClosureImpl, root_globals
from functools import reduce
import inspect


# When a Lambda is made into a FunctionImpl, we will
# cache it. This mostly avoids recomputing gradients,
# since FunctionImpl is the structure that stores
# pointers to them.
compile_cache: Dict[Lambda, 'FunctionImpl'] = {}
EnvT = Dict[Symbol, Any]


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
    def __init__(self, node: MyiaASTNode) -> None:
        self.node = node
        self.instructions: List[Instruction] = []
        self.process(self.node)

    def instr(self, name, node, *args) -> None:
        self.instructions.append(Instruction(name, node, *args))

    def process(self, node) -> None:
        # Dispatch to process_<node_type>
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
        self.debugger = debugger
        self.do_emit_events = emit_events
        # Current frame
        self.frame = VMFrame(self, code, list(envs))
        # Stack of previous frames (excludes current one)
        self.frames: List[VMFrame] = []
        if self.do_emit_events:
            self.emit_new_frame(self.frame)
        self.result = self.eval()

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
        """
        Pop ``nelems`` values from the stack and push a
        tuple of these values.
        """
        self.stack.append(tuple(self.take(nelems)))

    def instruction_closure(self, node) -> None:
        """
        Pop a tuple of arguments, and a function, and push
        ``ClosureImpl(fn, args)``
        """
        args = self.stack.pop()
        fn = self.stack.pop()
        clos = ClosureImpl(fn, args)
        self.stack.append(clos)

    def instruction_store(self, node, dest) -> None:
        """
        Pop a value and store it in the local environment
        under ``dest``. ``dest`` may be a Symbol or a tree
        of Symbols, represented as a Tuple.
        """
        value = self.stack.pop()

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
        """
        Push ``value`` on the stack.
        """
        self.stack.append(value)

    def instruction_lambda(self, node) -> None:
        """
        Create a FunctionImpl from the given node and push
        it on the stack.
        """
        fimpl = FunctionImpl(node, self.envs)
        self.stack.append(fimpl)

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
    itself (Parser stores a Lambda's global environment in its
    ``global_env`` field).
    """
    if isinstance(node, Lambda):
        parse_env = node.global_env
    assert parse_env is not None
    envs = (parse_env.bindings, root_globals)
    return run_vm(VMCode(node), *envs, debugger=debugger)


# from .runtime import EnvT
from . import runtime
runtime.run_vm = run_vm
runtime.VMCode = VMCode
