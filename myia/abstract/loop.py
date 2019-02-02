"""Core of the inference engine (not Myia-specific)."""

import asyncio
from contextvars import copy_context
from collections import deque

from ..dtype import Function, type_cloner_async, ismyiatype
from ..utils import Unification, Var, RestrictedVar, eprint, overload, \
    Overload


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

    def __init__(self, errtype):
        """Initialize an InferenceLoop."""
        self._todo = deque()
        self._tasks = []
        self._errors = []
        self._vars = []
        self.errtype = errtype

    def get_debug(self):
        """There is no debug mode."""
        return False

    def _resolve_var(self):
        varlist = [fut for fut in self._vars if not fut.done()]
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
            except self.errtype as e:
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
            raise AssertionError('call_exception_handler', ctx)

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
                exc = self.errtype(
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


def is_simple(x):
    from .data import AbstractScalar, TYPE
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
        self.set_result(self._resolve())

    def tie(self, other):
        assert isinstance(other, Pending)
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

    def _resolve(self):
        for e in self.equiv:
            if isinstance(e, Pending):
                if e.done():
                    return e.result()
            else:
                return e
        return self.default


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
        self.set_result(self.tentative)


async def find_coherent_result(v, fn):
    if isinstance(v, PendingFromList):
        results = set()
        for option in v.possibilities:
            results.add(await fn(option))
        if len(results) == 1:
            return results.pop()
    x = await v
    return await fn(x)


async def force_pending(v):
    if isinstance(v, Pending):
        return await v
    else:
        return v
