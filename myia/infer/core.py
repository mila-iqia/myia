"""Core of the inference engine (not Myia-specific)."""

import asyncio
from heapq import heappush, heappop

from .utils import InferenceError, DynamicMap, MyiaTypeError


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
        self._futures = []
        self._errors = []

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
                self._futures.append(fut)

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
        futs, self._futures = self._futures, []
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

            self._tie_dmaps(x, y, refs,)
            self._tie_dmaps(y, x, refs, hist=False)

        elif x == y:
            pass

        else:
            self.error_callback(
                MyiaTypeError(f'Type mismatch: {x} != {y}', refs=refs)
            )
