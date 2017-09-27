from typing import Dict, Callable, List, Any, Union, Tuple as TupType, Optional

import asyncio
from types import FunctionType
from ..stx import \
    MyiaASTNode, Location, Symbol, ValueNode, LambdaNode, maptup2, globals_pool
from ..lib import Closure, IdempotentMappable
from ..symbols import builtins, object_map, update_object_map
from ..util import EventDispatcher, BucheDb, HReprBase, buche
from functools import reduce
from ..impl.main import impl_bank
from ..parse import parse_function
from .vmutil import EvaluationEnv, EvaluationEnvCollection, Function, \
    VMCode, Instruction


# When a LambdaNode is made into a Function, we will
# cache it. This mostly avoids recomputing gradients,
# since Function is the structure that stores
# pointers to them.
compile_cache: Dict[LambdaNode, 'Function'] = {}
EnvT = Dict[Symbol, Any]
root_globals = impl_bank['interp']


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


class VM:
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
                 local_env: EnvT,
                 eval_env: EvaluationEnv,
                 controller = None) -> None:
        self.controller = controller
        self.do_emit_events = False
        # Current frame
        self.eval_env = eval_env
        self.frame = VMFrame(self, code, local_env, eval_env)
        # Stack of previous frames (excludes current one)
        self.frames: List[VMFrame] = []

    def eval(self, stop_on=True) -> Any:
        while True:
            try:
                # VMFrame does most of the work.
                if stop_on == True or \
                        stop_on is self.frame or \
                        stop_on is self.frame.focus:
                    stop_on = yield stop_on
                new_frame = self.frame.advance()
                if new_frame is not None:
                    # When the current frame gives us a new frame,
                    # we push the old one on the stack and start
                    # processing the new one.
                    if not self.frame.done():
                        # We push the current frame only if it's
                        # not done (this implement tail calls).
                        self.frames.append(self.frame)
                    self.frame = new_frame
            except StopIteration:
                # The result of a frame's evaluation is the value at
                # the top of its stack.
                rval = self.frame.top()
                if not self.frames:
                    # We are done!
                    return self.frame.top()
                else:
                    # We push the result on the previous frame's stack
                    # and we resume execution.
                    self.frame = self.frames.pop()
                    self.frame.push(rval)
            except Exception as exc:
                raise exc from None

    async def run_async(self):
        gen = self.eval()
        policy = gen.send(None)
        try:
            while True:
                policy = gen.send(await self.controller(self, policy))
        except StopIteration as exc:
            return exc.value        

    def run(self):
        if self.controller:
            return self.run_async()
        gen = self.eval()
        try:
            while True:
                next(gen)
        except StopIteration as exc:
            return exc.value


class VMFrame(HReprBase):
    """
    Computation frame. There is one frame for each Function
    called. A frame has its own stack, while the VM operates on
    a stack of VMFrames.

    Compute a frame's next instruction with ``advance()``,
    which may return a new VMFrame to the VM to compute something it
    needs, or throw StopIteration if it is done, in which case its
    return value is at the top of its stack.
    """
    def __init__(self,
                 vm: VM,
                 code: VMCode,
                 local_env: EnvT,
                 eval_env: EvaluationEnv) -> None:
        self.vm = vm
        self.code = code
        self.instructions = code.instructions
        # Program counter: index of the next instruction to execute.
        self.pc = 0
        # Environment to store local bindings.
        self.envs: List[EnvT] = [local_env, eval_env]
        self.local_env = local_env
        self.eval_env = eval_env
        self.stack: List[Any] = [None]
        # Node being executed, mostly for debugging.
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
        # self.stack += list(values)
        for value in values:
            value = self.eval_env.import_value(value)
            self.stack.append(value)

    def pop(self):
        return self.stack.pop()

    def current_instruction(self) -> Optional[Instruction]:
        """
        Get the next instruction to execute.

        Returns:
           * None if we are done.
           * The next instruction otherwise.
        """
        if self.pc >= len(self.instructions):
            return None
        instr = self.instructions[self.pc]
        return instr

    @property
    def focus(self):
        return self.rel_node(0)

    def rel_node(self, i=0):
        idx = self.pc + i
        if idx >= len(self.instructions) or idx < 0:
            return None
        instr = self.instructions[idx]
        return instr and instr.node

    def advance(self) -> Optional['VMFrame']:
        """
        Execute the next instruction.

        Returns:
            * ``None`` for most operations.
            * A ``VMFrame``. The VM should execute that frame
              and push its result to this frame before resuming
              execution.
        """
        instr = self.current_instruction()
        if not instr:
            raise StopIteration()
        else:
            self.pc += 1
            mname = 'instruction_' + instr.command
            method = getattr(self, mname)
            return method(instr.node, *instr.args)

    def instruction_reduce(self, node, nargs) -> Optional['VMFrame']:
        """
        * Pop ``nargs`` values from the stack, call them ``args``
        * Pop the next value, call it ``fn``
        * If ``fn`` is a ``Function``, we can run it with the VM.
          Make a new VMFrame for it and return it. This is important
          because we don't want to grow the Python stack.
        * Otherwise, it's a primitive. Call ``fn(*args)`` and push
          the result.
        """
        fn, *args = self.take(nargs + 1)
        if isinstance(fn, Function):
            bind: EnvT = {k: v for k, v in zip(fn.ast.args, args)}
            return self.__class__(self.vm, fn.code, bind, fn.eval_env)
        elif isinstance(fn, Closure):
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
        ``Closure(fn, args)``
        """
        fn, args = self.take(2)
        clos = Closure(fn, args)
        self.push(clos)

    def instruction_store(self, node, dest) -> None:
        """
        Pop a value and store it in the local environment
        under ``dest``. ``dest`` may be a Symbol or a tree
        of Symbols, represented as a Tuple.
        """
        value = self.pop()

        def store(dest, val):
            if isinstance(dest, Symbol):
                self.local_env[dest] = val
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
        Create a Function from the given node and push
        it on the stack.
        """
        fimpl = self.eval_env.compile(node)
        self.push(fimpl)

    def __hrepr__(self, H, hrepr):
        views = H.tabbedView['hrepr-VMFrame']()
        # env = {}
        # for e in reversed(self.envs):
        #     env.update(e)
        env = self.eval_env
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


eenvs = EvaluationEnvCollection(EvaluationEnv, root_globals,
                                globals_pool, VM, load)


def evaluate(node, controller=None):
    return eenvs.run_env(node, controller=controller)
