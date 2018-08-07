"""Core of the inference engine (not Myia-specific)."""

import asyncio
from heapq import heappush, heappop

from ..dtype import Array, List, Tuple, Function
from ..utils import TypeMap, Unification, Var

from .utils import InferenceError, DynamicMap, MyiaTypeError, ValueWrapper


class _TodoEntry:
    def __init__(self, order, handler):
        self.order = order
        self.handler = handler

    def __lt__(self, other):
        return self.order < other.order


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
        self._running = False
        self._todo = []
        self._debug = debug
        self._tasks = []
        self._errors = []
        self._vars = []

    def get_debug(self):
        """Not entirely sure what this does."""
        return self._debug

    def run_forever(self):
        """Run this loop until there is no more work to do."""
        self._running = True
        while self._todo and self._running:
            todo = heappop(self._todo)
            h = todo.handler
            if isinstance(h, asyncio.Handle):
                h._run()
            else:
                fut = asyncio.ensure_future(h, loop=self)
                self._tasks.append(fut)
        pending_vars = [fut for fut in self._vars if not fut.resolved()]
        if pending_vars:
            # If some literals weren't forced to a concrete type by some
            # operation, we sort by priority (i.e. floats first) and we
            # force the first one to take its default concrete type. Then
            # we resume the loop.
            pending_vars.sort(key=lambda x: -x.priority)
            v1, *self._vars = pending_vars
            v1.resolve_to_default()
            self.run_forever()

    def is_running(self):
        """Return whether the loop is running."""
        return self._running

    def is_closed(self):
        """Return whether the loop is closed."""
        return not self.is_running()

    def schedule(self, x, order=0):
        """Schedule a task with the given priority.

        A smaller value for `order` means higher priority.
        """
        # TODO: order argument may not be relevant anymore, so we might want to
        # remove it and simplify task sorting.
        heappush(self._todo, _TodoEntry(order, x))

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
        h = asyncio.Handle(callback, args, self)
        heappush(self._todo, _TodoEntry(0, h))
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

    def declare_equivalent(self, x, y, refs):
        """Declare that x and y should be equivalent.

        If an error occurs, the refs argument is to be packaged with it.
        """
        self.loop.schedule(self._process_equivalence(x, y, refs))

    def _tie_dmaps(self, src, dest, refs, hist=True):
        def evt(_, refs, res_src):
            res_dest = dest(*refs)
            self.declare_equivalent(res_src, res_dest, refs)
        src.on_result.register(evt, run_history=hist)

    async def _process_equivalence(self, x, y, refs):
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
            self.error_callback(
                MyiaTypeError(f'Type mismatch: {x} != {y}', refs=refs)
            )

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
        if self.done():
            self.result()._infvar.resolve_to(self.default)
        else:
            self.set_result(self.default)

    def resolve_to(self, value):
        """Resolve to the provided value."""
        if self.done():
            self.result()._infvar.resolve_to(value)
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
    return Array(await reify(t.elements))


@_reify_map.register(List)
async def _reify_List(t):
    return List(await reify(t.element_type))


@_reify_map.register(Tuple)
async def _reify_Tuple(t):
    return Tuple(await reify(t.elements))


@_reify_map.register(Function)
async def _reify_Function(t):
    return Function(await reify(t.arguments), await reify(t.retval))


@_reify_map.register(tuple)
async def _reify_tuple(v):
    li = [await reify(x) for x in v]
    return tuple(li)


@_reify_map.register(int)
async def _reify_int(v):
    return v


@_reify_map.register(type)
@_reify_map.register(object)
async def _reify_object(v):
    return v


async def reify(v):
    """Build a concrete value from v.

    All InferenceVars in v will be awaited on.
    """
    return await _reify_map[type(v)](v)
