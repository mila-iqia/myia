
from typing import List, Any, Dict
from ..interpret import VMCode, VMFrame, EnvT, \
    PrimitiveImpl, FunctionImpl, ClosureImpl, Instruction
from ..util import EventDispatcher, BucheDb, Keyword
from ..symbols import builtins
from ..impl.main import impl_bank
from itertools import product
from .dfa import DFA, ValueTrack, NeedsTrack
from ..stx import maptup2, Symbol, Tuple
from collections import defaultdict
from ..impl.flow_all import ANY, VALUE, ERROR, OPEN


compile_cache: Dict = {}
aroot_globals = impl_bank['abstract']
projector_set = set()


def wrap_abstract(v):
    if isinstance(v, AbstractValue):
        return v
    else:
        return AbstractValue(v)


def unwrap_abstract(v):
    if isinstance(v, AbstractValue):
        return v[VALUE]
    else:
        return v


class AbstractValue:
    def __init__(self, value):
        if isinstance(value, dict):
            self.values = value
        else:
            self.values = {VALUE: value}

    def acquire(self, proj):
        print(self, proj)
        if proj is VALUE:
            if VALUE in self.values:
                return self.values[VALUE]
            else:
                # raise ErrorValueException(self)
                raise Exception(f'No VALUE for {self}')
        else:
            assert proj
            return aroot_globals[proj](self)

    def __getitem__(self, proj):
        if proj not in self.values:
            res = self.acquire(proj)
            if proj is VALUE and isinstance(res, AbstractData):
                self.values = copy(res.values)
                if VALUE not in self.values:
                    raise ErrorValueException(self)
                return self[VALUE]
            else:
                self.values[proj] = res
        return self.values[proj]

    def __call__(self, proj):
        rval = self[proj]
        if proj is VALUE:
            return self
        else:
            return rval

    def __repr__(self):
        return repr(self.values)

    def __hash__(self):
        return hash(tuple(self.values.items()))

    def __eq__(self, other):
        return isinstance(other, AbstractValue) \
            and self.values == other.values


def load():
    from ..impl.impl_abstract import _
    from ..impl.proj_shape import _
    from ..impl.proj_type import _
    for p in ('shape', 'type'):
        projector_set.add(aroot_globals[builtins[p]])


class WrappedException(Exception):
    def __init__(self, error):
        super().__init__()
        self.error = error


class Escape(Exception):
    pass


class Fork:
    def __init__(self, paths):
        assert len(paths) > 0
        self.paths = paths

    def __str__(self):
        paths = ", ".join(map(str, self.paths))
        return f'Fork({paths})'


def find_projector(proj, fn):
    if proj is VALUE:
        return fn
    if isinstance(fn, PrimitiveImpl):
        try:
            return impl_bank['project'][proj][fn]
        except KeyError:
            raise Exception(f'Missing prim projector "{proj}" for {fn}.')
    # elif isinstance(fn, ClosureAImpl):
    #     return ClosureAImpl(find_projector(proj, fn.fn), fn.args)
    elif isinstance(fn, FunctionImpl):
        # def fn2(*args, **kw):
        #     return fn(*args, proj=proj)
        # return fn2
        return fn
    else:
        raise Exception(f'Cannot project "{proj}" with {fn}.')


class AVMFrame(VMFrame):
    def __init__(self,
                 vm,
                 code,
                 envs,
                 signature) -> None:
        from ..symbols import builtins
        from ..interpret import PrimitiveImpl
        super().__init__(vm, code, envs)
        self.signature = signature

    def take(self, n):
        taken = super().take(n)
        possibilities = list(product(*[x.paths if isinstance(x, Fork) else [x]
                                       for x in taken]))
        if len(possibilities) > 1:
            self.vm.checkpoint(possibilities[1:])
        return possibilities[0]

    def pop(self):
        return self.take(1)[0]

    def aux(self, node, fn, args, projs):
        nargs = len(args)
        args = [wrap_abstract(arg) for arg in args]
        instrs = []
        for proj in projs:
            pfn = find_projector(proj, fn)
            instrs.append(Instruction('push', node.fn, pfn))
            for arg in args:
                instrs.append(Instruction('push', None, arg))
            instrs.append(Instruction('reduce', node, nargs))
        instrs.append(Instruction('assemble', node, projs))
        vmc = VMCode(node, instrs)
        return self.__class__(self.vm, vmc, [], None)

    def instruction_store(self, node, dest) -> None:
        value = self.pop()
        if isinstance(dest, Tuple) and isinstance(value, AbstractValue):
            value = value[VALUE]

        def store(dest, val):
            if isinstance(dest, Symbol):
                self.envs[0][dest] = val
            else:
                raise TypeError(f'Cannot store into {dest}.')

        maptup2(store, dest, value)

    def instruction_assemble(self, node, projs):
        values = self.take(len(projs))
        value = AbstractValue({p: v for p, v in zip(projs, values)})
        self.push(value)

    def instruction_reduce(self, node, nargs):
        fn, *args = self.take(nargs + 1)
        if isinstance(fn, FunctionImpl):
            bind: EnvT = {k: v for k, v in zip(fn.ast.args, args)}
            sig = (fn, tuple(args))

            open, cached = self.vm.consult_cache(sig)
            if cached is None:
                return self.__class__(self.vm, fn.code, [bind] + fn.envs, sig)
            elif open:
                self.vm.checkpoint_on(sig)
                if not cached:
                    raise Escape()
                else:
                    self.push(Fork(cached))
            else:
                self.push(Fork(cached))

            # return self.__class__(self.vm, fn.code, [bind] + fn.envs, sig)
        elif isinstance(fn, ClosureImpl):
            self.push(fn.fn, *fn.args, *args)
            return self.instruction_reduce(node, nargs + len(fn.args))
        elif isinstance(fn, PrimitiveImpl):
            if nargs == 1:
                arg, = args
                if isinstance(arg, AbstractValue) and fn.name in arg.values:
                    self.push(arg[fn.name])
                    return None

            projs = self.vm.needs[node]
            if VALUE in projs or len(projs) == 0:
                value = fn(*args)
                self.push(value)
                return None
            else:
                return self.aux(node, fn, args, projs)
        else:
            raise TypeError(f'Cannot reduce on {fn}.')

    def copy(self, pc_offset=0):
        fr = AVMFrame(self.vm, self.code, self.envs, self.signature)
        fr.envs = list(self.envs)
        fr.envs[0] = {k: v for k, v in self.envs[0].items()}
        fr.stack = [s for s in self.stack]
        fr.pc = self.pc + pc_offset
        fr.focus = self.focus
        return fr


class AVM(EventDispatcher):
    def __init__(self,
                 code: VMCode,
                 needs: Dict,
                 *envs: EnvT,
                 debugger: BucheDb = None,
                 emit_events=True) -> None:
        super().__init__(self, emit_events)
        self.results_cache: Dict = {}
        self.open_cache: Dict = {}
        self.compile_cache = compile_cache
        self.needs = needs
        self.debugger = debugger
        self.do_emit_events = emit_events
        # Current frame
        self.frame = AVMFrame(self, code, list(envs), None)
        # Stack of previous frames (excludes current one)
        self.frames: List[AVMFrame] = []
        if self.do_emit_events:
            self.emit_new_frame(self.frame)
        self.result = self.eval()
        self.checkpoints: List = []
        self.checkpoints_on: Dict = defaultdict(list)
        self.sig_stack: List = []

    def consult_cache(self, key):
        try:
            open = self.open_cache[key]
            results = self.results_cache[key]
            return open, results
        except ValueError:
            results = None
            return False, None
        except KeyError:
            self.open_cache[key] = True
            self.results_cache[key] = set()
            return False, None

    def add_cache(self, sig, value):
        if value in self.results_cache[sig]:
            return
        self.results_cache[sig].add(value)
        for (frs, fr, sig_stack) in self.checkpoints_on[sig]:
            self.checkpoints.append((frs, fr, sig_stack, [[value]]))

    def checkpoint(self, paths):
        # fr = AVMFrame(self, self.frame.code, self.frame.envs)
        # fr.pc = self.frame.pc - 1
        fr = self.frame.copy(-1)
        frs = [f.copy() for f in self.frames]
        self.checkpoints.append((frs, fr, set(self.sig_stack), paths))

    def checkpoint_on(self, sig):
        fr = self.frame.copy()
        frs = [f.copy() for f in self.frames]
        self.checkpoints_on[sig].append((frs, fr, set(self.sig_stack)))

    def evaluate(self, lbda):
        parse_env = lbda.global_env
        assert parse_env is not None
        envs = (parse_env.bindings, aroot_globals)
        fn, = list(run_avm(VMCode(lbda), *envs))
        return fn

    def go(self) -> Any:
        while True:
            try:
                # AVMFrame does most of the work.
                new_frame = self.frame.next()
                if new_frame is not None:
                    if new_frame.signature:
                        self.sig_stack.append(new_frame.signature)
                    # When the current frame gives us a new frame,
                    # we push the old one on the stack and start
                    # processing the new one.
                    if True or not self.frame.done():
                        # We push the current frame only if it's
                        # not done (this implement tail calls).
                        self.frames.append(self.frame)
                    self.frame = new_frame  # type: ignore
                    if self.do_emit_events:
                        self.emit_new_frame(self.frame)
            except Escape:
                return None
            except StopIteration:
                # The result of a frame's evaluation is the value at
                # the top of its stack.
                rval = self.frame.top()
                if isinstance(rval, Fork):
                    self.frame.pop()
                    self.checkpoint([[p] for p in rval.paths])
                    raise Escape()

                sig = self.frame.signature
                if sig is not None:
                    # self.sig_stack.pop()
                    # if all(sig not in sigs
                    #        for _, _, sigs, _ in self.checkpoints):
                    #     self.open_cache[sig] = False
                    self.add_cache(sig, rval)

                if not self.frames:
                    # We are done!
                    return rval
                else:
                    # We push the result on the previous frame's stack
                    # and we resume execution.
                    self.frame = self.frames.pop()
                    self.frame.stack.append(rval)
            # except Exception as exc:
            #     if self.do_emit_events:
            #         self.emit_error(exc)
            #     raise exc from None
            except WrappedException as exc:
                return {ERROR: exc.error}

    def eval(self) -> Any:
        while True:
            rval = self.go()
            if rval is not None:
                yield rval
            if self.checkpoints:
                frames, frame, sig_stack, paths = self.checkpoints.pop()
                p, *rest = paths
                self.frames = [f.copy() for f in frames]
                self.frame = frame.copy()
                self.frame.push(*p)
                if rest:
                    self.checkpoints.append((frames, frame, sig_stack, rest))
            else:
                return


def run_avm(code: VMCode,
            *binding_groups: EnvT,
            debugger: BucheDb = None) -> Any:
    """
    Execute the VM on the given code.
    """
    return AVM(code, *binding_groups, debugger=debugger).result


def abstract_evaluate(lbda, args, proj=None):
    load()
    parse_env = lbda.global_env
    assert parse_env is not None
    envs = (parse_env.bindings, aroot_globals)

    # d = DFA([ValueTrack, lambda dfa: NeedsTrack(dfa, [proj or VALUE])],
    #         parse_env)
    d = DFA([ValueTrack, lambda dfa: NeedsTrack(dfa, [])],
            parse_env)
    d.visit(lbda)
    d.propagate(lbda.body, 'needs', proj or VALUE)

    fn, = list(run_avm(VMCode(lbda), *envs))
    return run_avm(fn.code,
                   d.values[d.tracks['needs']],
                   {s: arg for s, arg in zip(lbda.args, args)},
                   *fn.envs)
