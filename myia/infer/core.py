"""Core of the inference engine (not Myia-specific)."""

import asyncio
from contextvars import copy_context
from collections import deque

from ..dtype import Function, type_cloner_async, ismyiatype
from ..utils import Unification, Var, RestrictedVar, eprint, overload, \
    Overload

from .utils import InferenceError, MyiaTypeError


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


class InferenceTask(asyncio.Task):
    def __init__(self, coro, loop, key=None):
        super().__init__(coro, loop=loop)
        self.key = key


class InferenceLoop(asyncio.AbstractEventLoop):
    """EventLoop implementation for use with the inferrer.

    This event loop doesn't allow scheduling tasks and callbacks with
    `call_later` or `call_at`, which means the `timeout` argument to methods
    like `wait` will not work. `run_forever` will stop when it has exhausted
    all work there is to be done. This means `run_until_complete` may finish
    before it can evaluate the future, which suggests an infinite loop.
    """

    def __init__(self):
        """Initialize an InferenceLoop."""
        self._todo = deque()
        self._tasks = []
        self._errors = []
        self._vars = []
        # This is used by InferenceVar and EquivalenceChecker:
        self.equiv = {}

    def get_debug(self):
        """There is no debug mode."""
        return False

    def _resolve_var(self):
        varlist = [fut for fut in self._vars if not fut.resolved()]
        later = [fut for fut in varlist if fut.priority() is None]
        varlist = [fut for fut in varlist if fut.priority() is not None]
        self._vars = later
        if not varlist:
            return False
        varlist.sort(key=lambda x: x.priority())
        found = False
        while varlist:
            v1 = varlist.pop()
            try:
                v1.force_resolve()
            except InferenceError as e:
                self._errors.append(e)
            else:
                found = True
                break
        self._vars += varlist
        return found

    def run_forever(self):
        """Run this loop until there is no more work to do."""
        while True:
            while self._todo:
                h = self._todo.popleft()
                h._run()
            # If some literals weren't forced to a concrete type by some
            # operation, we sort by priority (i.e. floats first) and we
            # force the first one to take its default concrete type. Then
            # we resume the loop.
            if not self._resolve_var():
                break

    def call_exception_handler(self, ctx):
        if 'exception' in ctx:
            raise ctx['exception']
        else:
            raise Exception('????')

    def schedule(self, x, context_map=None):
        """Schedule a task."""
        if context_map:
            ctx = copy_context()
            ctx.run(lambda: [k.set(v) for k, v in context_map.items()])
            fut = ctx.run(asyncio.ensure_future, x, loop=self)
        else:
            fut = asyncio.ensure_future(x, loop=self)
        if isinstance(fut, InferenceTask):
            fut.key = context_map
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
        return InferenceTask(coro, loop=self)

    def create_pending(self, resolve, priority, parents=()):
        pending = Pending(resolve=resolve, priority=priority, loop=self)
        self._vars.append(pending)
        return pending

    def create_pending_from_list(self, poss, dflt, priority):
        pending = PendingFromList(poss, dflt, priority, loop=self)
        self._vars.append(pending)
        return pending

    def create_pending_tentative(self, tentative):
        pending = PendingTentative(tentative=tentative, loop=self)
        self._vars.append(pending)
        return pending


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

    def clear(self):
        """Clear the cache completely."""
        self.cache.clear()


def is_simple(x):
    from ..abstract.base import AbstractScalar, TYPE
    if isinstance(x, Pending):
        return x.is_simple()
    if isinstance(x, AbstractScalar):
        return is_simple(x.values[TYPE])
    elif ismyiatype(x):
        return True
    else:
        return False


class Pending(asyncio.Future):
    def __init__(self, resolve, priority, loop):
        super().__init__(loop=loop)
        self.priority = priority
        if resolve is not None:
            self._resolve = resolve
        self.equiv = {self}

    def is_simple(self):
        return False

    def force_resolve(self):
        """Resolve to the default value."""
        self.resolve_to(self._resolve())

    def resolve_to(self, value):
        self.set_result(value)
        # TODO: check whether this makes _resolve_from_equiv unneeded
        # if is_simple(self):
        #     for e in self.equiv:
        #         if isinstance(e, Pending) and not e.done():
        #             e.set_result(value)

    def resolved(self):
        return self.done()

    def tie(self, other):
        if isinstance(other, Pending):
            self.equiv |= other.equiv
            e = self.equiv
            for p in self.equiv:
                p.equiv = e


class PendingFromList(Pending):
    def __init__(self, possibilities, default, priority, loop):
        super().__init__(
            resolve=None,
            priority=priority,
            loop=loop
        )
        self.default = default
        self.possibilities = possibilities

    def is_simple(self):
        return all(is_simple(p) for p in self.possibilities)

    def _resolve_from_equiv(self):
        for e in self.equiv:
            if isinstance(e, Pending):
                if e.done():
                    return e.result()
            else:
                return e
        return None

    def _resolve(self):
        x = self._resolve_from_equiv()
        return self.default if x is None else x

    def tie(self, other):
        super().tie(other)
        x = self._resolve_from_equiv()
        if x is not None:
            self.resolve_to(x)


class PendingTentative(Pending):
    def __init__(self, tentative, loop):
        super().__init__(
            resolve=None,
            priority=self._priority,
            loop=loop
        )
        self.tentative = tentative

    def _priority(self):
        if isinstance(self.tentative, Pending):
            assert False  # TODO
            return None
        else:
            return -1001

    def force_resolve(self):
        """Resolve to the tentative value."""
        self.resolve_to(self.tentative)


async def find_coherent_result(v, fn):
    if isinstance(v, PendingFromList):
        results = set()
        for option in v.possibilities:
            results.add(await fn(option))
        if len(results) == 1:
            return results.pop()
    x = await v
    return await fn(x)


async def reify(v):
    if isinstance(v, Pending):
        return await v
    else:
        return v
