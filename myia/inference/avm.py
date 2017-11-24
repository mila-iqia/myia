
from typing import List, Any, Dict
from ..interpret import VMCode, VMFrame, EnvT, \
    Primitive, Function, Closure, Instruction, \
    EvaluationEnv, EvaluationEnvCollection
from ..util import EventDispatcher
from ..symbols import builtins
from ..impl.main import impl_bank
from itertools import product
from .dfa import DFA, ValueTrack, NeedsTrack
from ..stx import maptup2, Symbol, TupleNode, LambdaNode, globals_pool
from collections import defaultdict
from ..lib import ANY, VALUE, ERROR
from .types import Type


aroot_globals = impl_bank['abstract']
projector_set = set()
max_depth = 5


class SetDepth:
    def __init__(self, depth):
        self.prev_depth = max_depth
        self.depth = depth

    def __enter__(self):
        global max_depth
        max_depth = self.depth

    def __exit__(self, excv, exct, exctb):
        global max_depth
        max_depth = self.prev_depth


def wrap_abstract(v):
    if isinstance(v, AbstractValue):
        return v
    else:
        return AbstractValue(v)


def unwrap_abstract(v):
    if isinstance(v, AbstractValue):
        if VALUE in v.values:
            return unwrap_abstract(v[VALUE])
        else:
            return ANY
    else:
        return v


class AbstractValue:
    def __init__(self, value, depth=0):
        self.depth = depth
        if depth > max_depth:
            value = ANY
        if isinstance(value, dict):
            self.values = value
        else:
            self.values = {VALUE: value}
        if VALUE in self.values:
            while isinstance(self[VALUE], AbstractValue):
                v = self[VALUE]
                self.depth = max(v.depth, self.depth)
                self.values[VALUE] = v[VALUE]

    def __getitem__(self, proj):
        return self.values[proj]

    def __call__(self, proj):
        rval = self[proj]
        if proj is VALUE:
            return self
        else:
            return rval

    def __repr__(self):
        return repr(self.values)

    def __hrepr__(self, H, hrepr):
        return hrepr(self.values)

    def __hash__(self):
        return hash(tuple(self.values.items()))

    def __eq__(self, other):
        return isinstance(other, AbstractValue) \
            and self.values == other.values


def load():
    from ..impl.impl_interp import _
    from ..impl.impl_abstract import _
    from ..impl.proj_shape import _
    from ..impl.proj_type import _
    for p in ('shape', 'type'):
        projector_set.add(aroot_globals[builtins[p]])


class WrappedException(Exception):
    def __init__(self, error):
        super().__init__(error)
        self.error = error

    def __hash__(self):
        return hash(str(self.error))

    def __eq__(self, other):
        return type(self) is type(other) \
            and str(self.error) == str(other.error)


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
    if isinstance(fn, Primitive):
        try:
            return impl_bank['project'][proj][fn]
        except KeyError:
            raise Exception(f'Missing prim projector "{proj}" for {fn}.')
    elif isinstance(fn, Function):
        return fn
    else:
        raise Exception(f'Cannot project "{proj}" with {fn}.')


class AVMFrame(VMFrame):
    def __init__(self,
                 vm,
                 code,
                 local_env,
                 universe,
                 signature=None) -> None:
        from ..symbols import builtins
        from ..interpret import Primitive
        super().__init__(vm, code, local_env, universe)
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

    def push(self, *values):
        def ann(v):
            if isinstance(v, AbstractValue):
                for proj, value in v.values.items():
                    if proj != VALUE:
                        self.vm.annotate(proj, value)
            elif isinstance(v, Fork):
                for vv in v.paths:
                    ann(vv)

        # values = [value if isinstance(value, AbstractValue)
        #           else AbstractValue(value) for value in values]
        super().push(*values)
        for v in values:
            ann(v)

    def push_no_annotate(self, *values):
        super().push(*values)

    def aux(self, node, fn, args, projs):
        if isinstance(node, TupleNode):
            nfn = builtins.mktuple
        else:
            nfn = node.fn
        nargs = len(args)
        args = [wrap_abstract(arg) for arg in args]
        instrs = []
        for proj in projs:
            pfn = find_projector(proj, fn)
            instrs.append(Instruction('push', nfn, pfn))
            for arg in args:
                instrs.append(Instruction('push', None, arg))
            instrs.append(Instruction('reduce', node, nargs, False))
        instrs.append(Instruction('assemble', node, projs))
        vmc = VMCode(node, instrs)
        return self.__class__(self.vm, vmc, {}, self.universe, None)

    def instruction_closure(self, node) -> None:
        fn, args = self.take(2)
        fn = unwrap_abstract(fn)
        args = unwrap_abstract(args)
        clos = Closure(fn, args)
        self.stack.append(clos)

    def instruction_store(self, node, dest) -> None:
        def store(dest, val):
            if isinstance(dest, Symbol):
                self.envs[0][dest] = val
            else:
                raise TypeError(f'Cannot store into {dest}.')

        def spread_error(x):
            if isinstance(x, TupleNode):
                for y in x.values:
                    spread_error(y)
            else:
                store(x, err)

        value = self.pop()
        if isinstance(dest, TupleNode) and isinstance(value, AbstractValue):
            if ERROR in value.values:
                err = value[ERROR]
                spread_error(dest)
                return
            elif VALUE in value.values:
                value = value[VALUE]
            else:
                err = AbstractValue({ERROR: WrappedException("No VALUE.")})
                spread_error(dest)
                return

        maptup2(store, dest, value)

    def instruction_assemble(self, node, projs):
        values = self.take(len(projs))
        value = AbstractValue({p: v for p, v in zip(projs, values)})
        self.push(value)

    def instruction_reduce(self, node, nargs, has_projs=True):
        fn, *args = self.take(nargs + 1)
        fn = unwrap_abstract(fn)
        if isinstance(fn, Function):
            bind: EnvT = {k: v for k, v in zip(fn.ast.args, args)}
            sig = (fn, tuple(args))

            open, cached = self.vm.consult_cache(sig)
            if cached is None:
                return self.__class__(self.vm, fn.code, bind,
                                      fn.universe, sig)
            elif open:
                self.vm.checkpoint_on(sig)
                if not cached:
                    raise Escape()
                else:
                    self.push(Fork(cached))
            else:
                self.push(Fork(cached))

        elif isinstance(fn, Closure):
            self.push_no_annotate(fn.fn, *fn.args, *args)
            return self.instruction_reduce(node, nargs + len(fn.args))

        elif isinstance(fn, Primitive):
            if nargs == 1:
                arg, = args
                if isinstance(arg, AbstractValue) and fn.name in arg.values:
                    self.push(arg[fn.name])
                    return None

            projs = self.vm.needs[node] if has_projs else set()
            if projs == {VALUE} or len(projs) == 0:
                try:
                    value = fn(*args)
                except WrappedException as exc:
                    value = AbstractValue({ERROR: exc})
                self.push(value)
                return None
            else:
                return self.aux(node, fn, args, projs)
        elif fn is ANY:
            self.push(ANY)
        else:
            raise TypeError(f'Cannot reduce on {fn}.')

    def copy(self, pc_offset=0):
        local_env = {**self.local_env}
        universe = self.universe
        fr = AVMFrame(self.vm, self.code,
                      local_env, universe, self.signature)
        fr.stack = [s for s in self.stack]
        fr.pc = self.pc + pc_offset
        return fr


class AVM(EventDispatcher):
    def __init__(self,
                 code: VMCode,
                 local_env: EnvT,
                 universe: EnvT,
                 signature=None,
                 needs: Dict = {},
                 projs=None,
                 emit_events=True) -> None:
        super().__init__(self, emit_events)
        self.results_cache: Dict = {}
        self.open_cache: Dict = {}
        self.needs = needs
        self.do_emit_events = emit_events
        # Current frame
        self.universe = universe
        self.frame = AVMFrame(self, code, local_env, universe, signature)
        # Stack of previous frames (excludes current one)
        self.frames: List[AVMFrame] = []
        if self.do_emit_events:
            self.emit_new_frame(self.frame)
        self.checkpoints: List = []
        self.checkpoints_on: Dict = defaultdict(list)
        self.sig_stack: List = []
        self.annotations: Dict = \
            defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    def annotate(self, track, value, node=None):
        # path = [f.focus for f in self.frames]
        node = node or self.frame.focus
        path = tuple(f.signature[0].ast.ref or '?' if f.signature else '?'
                     for f in self.frames + [self.frame])
        self.annotations[node][track][path].add(value)

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
        if sig not in self.results_cache:
            return
        if value in self.results_cache[sig]:
            return
        self.results_cache[sig].add(value)
        for (frs, fr, sig_stack) in self.checkpoints_on[sig]:
            self.checkpoints.append((frs, fr, sig_stack, [[value]], True))

    def checkpoint(self, paths, step_back=-1):
        fr = self.frame.copy(step_back)
        frs = [f.copy() for f in self.frames]
        self.checkpoints.append((frs, fr, set(self.sig_stack), paths, False))

    def checkpoint_on(self, sig):
        fr = self.frame.copy()
        frs = [f.copy() for f in self.frames]
        self.checkpoints_on[sig].append((frs, fr, set(self.sig_stack)))

    def go(self) -> Any:
        while True:
            try:
                # AVMFrame does most of the work.
                new_frame = self.frame.advance()
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
                    self.checkpoint([[p] for p in rval.paths], 0)
                    return None

                sig = self.frame.signature
                if sig is not None:
                    # TODO: reintegrate this code to close open
                    # function evaluations when possible.
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
                    self.frame.push(rval)
            # except Exception as exc:
            #     if self.do_emit_events:
            #         self.emit_error(exc)
            #     raise exc from None
            except WrappedException as exc:
                return AbstractValue({ERROR: exc.error})

    def eval(self) -> Any:
        while True:
            rval = self.go()
            if rval is not None:
                yield rval
            if self.checkpoints:
                frames, frame, sig_stack, paths, annotate = \
                    self.checkpoints.pop()
                p, *rest = paths
                self.frames = [f.copy() for f in frames]
                self.frame = frame.copy()
                if annotate:
                    self.frame.push(*p)
                else:
                    self.frame.push_no_annotate(*p)
                if rest:
                    self.checkpoints.append((frames, frame, sig_stack, rest))
            else:
                return

    def run(self) -> Any:
        self.result = self.eval()
        return self.result


class AEvaluationEnv(EvaluationEnv):
    def __init__(self, primitives, pool, config={}):
        super().__init__(primitives, pool, config)
        projs = config['projs']
        self.projs = projs
        self.dfa = DFA([ValueTrack, self.needs_track], self.pool)
        self.config['needs'] = self.dfa.values[self.dfa.tracks['needs']]

    def setup(self):
        load()

    def vm(self, code, local_env):
        return AVM(code, local_env, self, **self.config)

    def vmc(self, ast, instructions=None):
        return VMCode(ast, instructions, use_new_ir=False)

    def needs_track(self, dfa):
        return NeedsTrack(dfa, self.projs)

    def evaluate(self, lbda):
        self.setup()
        self.dfa.visit(lbda)
        fn = self[lbda]
        return unwrap_abstract(fn)

    def import_value(self, v):
        accepted_types = (AbstractValue, Fork, WrappedException, Type)
        if isinstance(v, accepted_types) or v is ANY:
            return v
        else:
            return super().import_value(v)


eenvs = EvaluationEnvCollection(AEvaluationEnv, aroot_globals,
                                globals_pool, cache=False)


def abstract_evaluate(node, proj=None):
    if not proj:
        proj = (VALUE,)
    elif not isinstance(proj, (list, tuple)):
        proj = (proj,)
    elif isinstance(proj, list):
        proj = tuple(proj)
    return eenvs.run_env(node, projs=proj)


avm_eenv = AEvaluationEnv
