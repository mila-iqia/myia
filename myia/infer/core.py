"""Core of the inference engine (not Myia-specific)."""

import asyncio
from contextvars import copy_context
from collections import deque

from ..dtype import Array, List, Tuple, Function, TypeMeta
from ..utils import TypeMap, Unification, Var, RestrictedVar, eprint

from .utils import InferenceError, DynamicMap, MyiaTypeError, ValueWrapper


class MyiaTypeMismatchError(MyiaTypeError):
    """Error where two values should have the same type, but don't."""

    def __init__(self, type1, type2, *, refs):
        """Initialize a MyiaTypeMismatchError."""
        msg = f'{type1} != {type2}'
        super().__init__(msg, refs=refs)
        self.type1 = type1
        self.type2 = type2


class MyiaFunctionMismatchError(MyiaTypeMismatchError):
    """Error where two functions should have the same type, but don't."""

    def __init__(self, type1, type2, *, refs):
        """Initialize a MyiaFunctionMismatchError."""
        super().__init__(type1, type2, refs=refs)

    def print_tb_end(self, fn_ctx, args_ctx, is_prim):
        """Print the error message at the end of a traceback."""
        m = "Two functions with incompatible return types may be called here"
        eprint(f'{type(self).__qualname__}: {m}: {self.message}')


class InferenceLoop(asyncio.AbstractEventLoop):
    """EventLoop implementation for use with the inferrer.

    This event loop doesn't allow scheduling tasks and callbacks with
    `call_later` or `call_at`, which means the `timeout` argument to methods
    like `wait` will not work. `run_forever` will stop when it has exhausted
    all work there is to be done. This means `run_until_complete` may finish
    before it can evaluate the future, which suggests an infinite loop.
    """

    def __init__(self, debug=False):
        """Initialize an InferenceLoop."""
        self._todo = deque()
        self._debug = debug
        self._tasks = []
        self._errors = []
        self._vars = []

    def get_debug(self):
        """Not entirely sure what this does."""
        return self._debug

    def run_forever(self):
        """Run this loop until there is no more work to do."""
        while True:
            while self._todo:
                h = self._todo.popleft()
                h._run()
            pending_vars = [fut for fut in self._vars if not fut.resolved()]
            if pending_vars:
                # If some literals weren't forced to a concrete type by some
                # operation, we sort by priority (i.e. floats first) and we
                # force the first one to take its default concrete type. Then
                # we resume the loop.
                pending_vars.sort(key=lambda x: -x.priority)
                v1, *self._vars = pending_vars
                try:
                    v1.resolve_to_default()
                except InferenceError as e:
                    self._errors.append(e)
                else:
                    continue  # pragma: no cover
            break

    def schedule(self, x, context_map=None):
        """Schedule a task."""
        if context_map:
            ctx = copy_context()
            ctx.run(lambda: [k.set(v) for k, v in context_map.items()])
            fut = ctx.run(asyncio.ensure_future, x, loop=self)
        else:
            fut = asyncio.ensure_future(x, loop=self)
        self._tasks.append(fut)
        return fut

    def collect_errors(self):
        """Return a collection of all exceptions from all futures."""
        futs, self._tasks = self._tasks, []
        errors, self._errors = self._errors, []
        for fut in futs:
            if fut.done():
                exc = fut.exception()
            else:
                exc = InferenceError(
                    f'Could not run inference to completion.'
                    ' There might be an infinite loop in the program'
                    ' which prevents type inference from working.',
                    refs=[]
                )
            if exc is not None:
                errors.append(exc)
        return errors

    def call_soon(self, callback, *args, context=None):
        """Call the given callback as soon as possible."""
        h = asyncio.Handle(callback, args, self, context=context)
        self._todo.append(h)
        return h

    def call_later(self, delay, callback, *args, context=None):
        """Not supported."""
        raise NotImplementedError(
            '_InferenceLoop does not allow timeouts or time-based scheduling.'
        )

    def call_at(self, when, callback, *args, context=None):
        """Not supported."""
        raise NotImplementedError(
            '_InferenceLoop does not allow time-based scheduling.'
        )

    def create_task(self, coro):
        """Create a task from the given coroutine."""
        return asyncio.Task(coro, loop=self)

    def create_future(self):
        """Create a Future using this loop."""
        return asyncio.Future(loop=self)

    def create_var(self, var, default, priority=0):
        """Create an InferenceVar running on this loop."""
        v = InferenceVar(var, default, priority, loop=self)
        self._vars.append(v)
        return v


class EvaluationCache:
    """Key/value store where keys are associated to Futures.

    Attributes:
        cache: The cache.
        loop: The InferenceLoop for async evaluation.
        keycalc: An async function that takes a key and returns
            the value associated to that key.

    """

    def __init__(self, loop, keycalc):
        """Initialize an EvaluationCache."""
        self.cache = {}
        self.loop = loop
        self.keycalc = keycalc

    def get(self, key):
        """Get the future associated to the key."""
        if key not in self.cache:
            self.set(key, self.keycalc(key))
        return self.cache[key]

    def set(self, key, coro):
        """Associate a key to a coroutine."""
        self.cache[key] = self.loop.create_task(coro)

    def set_value(self, key, value):
        """Associate a key to a value.

        This will wrap the value in a Future.
        """
        fut = asyncio.Future(loop=self.loop)
        fut.set_result(value)
        self.cache[key] = fut


class EquivalenceChecker:
    """Handle equivalence between values."""

    def __init__(self, loop, error_callback):
        """Initialize the EquivalenceChecker."""
        self.loop = loop
        self.error_callback = error_callback
        self.unif = Unification()
        self.equiv = {}

    def declare_equivalent(self, x, y, refs, error_callback=None):
        """Declare that x and y should be equivalent.

        If an error occurs, the refs argument is to be packaged with it.
        """
        coro = self._process_equivalence(x, y, refs, error_callback)
        self.loop.schedule(coro)

    def _tie_dmaps(self, src, dest, refs, hist=True):
        def evt(_, argrefs, res_src):
            async def acb(err):
                # TODO: this should fetch the appropriate track, not
                # necessarily 'type'
                argt = [await ref['type'] for ref in argrefs]
                t1 = Function[argt, err.type1]
                t2 = Function[argt, err.type2]
                err = MyiaFunctionMismatchError(t1, t2, refs=err.refs)
                self.error_callback(err)

            def cb(err):
                self.loop.schedule(acb(err))

            res_dest = dest(*argrefs)
            self.declare_equivalent(res_src, res_dest, refs, cb)
        src.on_result.register(evt, run_history=hist)

    async def _process_equivalence(self, x, y, refs, error_callback=None):
        if error_callback is None:
            error_callback = self.error_callback

        if hasattr(x, '__await__'):
            x = await x
        if hasattr(y, '__await__'):
            y = await y

        if isinstance(x, DynamicMap) and isinstance(y, DynamicMap):
            if x.provably_equivalent(y):
                return

            self._tie_dmaps(x, y, refs)
            self._tie_dmaps(y, x, refs)

        elif x == y or self.merge(x, y):
            pass

        else:
            error_callback(MyiaTypeMismatchError(x, y, refs=refs))

    def merge(self, x, y):
        """Merge the two values/variables x and y."""
        res = self.unif.unify(x, y, self.equiv)
        if res is None:
            return False
        self.equiv = res
        for var, value in self.equiv.items():
            iv = var._infvar
            if isinstance(value, Var) and not hasattr(value, '_infvar'):
                # Unification may create additional variables
                self.loop.create_var(value, iv.default, iv.priority)
            if not iv.done():
                iv.set_result(value)
        return True

    async def assert_same(self, *futs, refs=[]):
        """Assert that all refs have the same value on the given track."""
        # We wait only for the first future to complete
        done, pending = await asyncio.wait(
            futs,
            loop=self.loop,
            return_when=asyncio.FIRST_COMPLETED
        )

        # We must now tell equiv that all remaining futures must return the
        # same thing as the first one. This will essentially schedule a
        # bunch of tasks to wait for the remaining futures and verify that
        # they match. See EquivalenceChecker.
        main = done.pop()
        for fut in done | pending:
            self.declare_equivalent(fut, main, refs)

        # We return the first result immediately
        return main.result()


class InferenceVar(asyncio.Future):
    """Hold a Var that stands in for an inference result.

    This is a Future which can be awaited. Await on the `reify` function to
    get the concrete inference value for this InferenceVar.

    Arguments:
        var: A Var instance that can be unified with other Vars and values.
        default: The concrete value this InferenceVar will resolve to if
            the unification process fails to force a value.
        priority: When multiple InferenceVars have to be forced to their
            default values, those with higher priority are processed first.
        loop: The InferenceLoop this InferenceVar is attached to.
    """

    def __init__(self, var, default, priority, loop):
        """Initialize an InferenceVar."""
        super().__init__(loop=loop)
        self.var = var
        self.default = default
        self.priority = priority
        self.__var__ = var
        var._infvar = self

    def resolve_to_default(self):
        """Resolve to the default value."""
        default = self.default
        if default is None:
            if self.done():
                self.result()._infvar.resolve_to_default()
                return
            elif isinstance(self.var, RestrictedVar) \
                    and len(self.var.legal_values) == 1:
                for val in self.var.legal_values:
                    self.set_result(val)
                    return
            raise InferenceError('Could not resolve a variable type.', refs=[])
        else:
            if self.done():
                self.result()._infvar.resolve_to(default)
            else:
                self.set_result(default)

    def resolve_to(self, value):
        """Resolve to the provided value."""
        if self.done():
            # This used to happen, not sure how to trigger now.
            self.result()._infvar.resolve_to(value)  # pragma: no cover
        else:
            self.set_result(value)

    def resolved(self):
        """Whether this was resolved to a concrete value."""
        if not self.done():
            return False
        else:
            res = self.result()
            if isinstance(res, Var):
                return res._infvar.resolved()
            else:
                return True

    async def __reify__(self):
        """Map this InferenceVar to a concrete value."""
        return await reify(await self)


_reify_map = TypeMap(discover=lambda cls: getattr(cls, '__reify__', None))


@_reify_map.register(ValueWrapper)
async def _reify_ValueWrapper(x):
    return await reify(x.value)


@_reify_map.register(Var)
async def _reify_Var(v):
    return await reify(await v._infvar)


@_reify_map.register(Array)
async def _reify_Array(t):
    return Array[await reify(t.elements)]


@_reify_map.register(List)
async def _reify_List(t):
    return List[await reify(t.element_type)]


@_reify_map.register(Tuple)
async def _reify_Tuple(t):
    return Tuple[await reify(t.elements)]


@_reify_map.register(Function)
async def _reify_Function(t):
    return Function[await reify(t.arguments), await reify(t.retval)]


@_reify_map.register(tuple)
async def _reify_tuple(v):
    li = [await reify(x) for x in v]
    return tuple(li)


@_reify_map.register(int)
async def _reify_int(v):
    return v


@_reify_map.register(TypeMeta)
async def _reify_tmeta(v):
    if v.is_generic():
        return v
    else:
        return await _reify_map[v](v)


@_reify_map.register(type)
@_reify_map.register(object)
async def _reify_object(v):
    return v


async def reify(v):
    """Build a concrete value from v.

    All InferenceVars in v will be awaited on.
    """
    return await _reify_map[type(v)](v)
