"""Asyncio-related code for inference."""

import asyncio
from contextvars import copy_context
from collections import deque

from ..dtype import ismyiatype


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
        """Try to forcefully resolve one variable to resume execution.

        For example, if the code contains the literal 1.0, we let its type
        be determined by variables it interacts with, but if there is nothing
        else to do, we may force it to Float[64].
        """
        # Filter out all done tasks
        varlist = [fut for fut in self._vars if not fut.done()]
        # Filter out priority-less tasks, which cannot be forced
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
            self._errors.append(ctx['exception'])
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
        return asyncio.Task(coro, loop=self)

    def create_pending(self, resolve, priority):
        """Create a Pending associated to this loop."""
        pending = Pending(resolve=resolve, priority=priority, loop=self)
        self._vars.append(pending)
        return pending

    def create_pending_from_list(self, poss, dflt, priority):
        """Create a PendingFromList associated to this loop."""
        pending = PendingFromList(poss, dflt, priority, loop=self)
        self._vars.append(pending)
        return pending

    def create_pending_tentative(self, tentative):
        """Create a PendingTentative associated to this loop."""
        pending = PendingTentative(tentative=tentative, loop=self)
        self._vars.append(pending)
        return pending


def is_simple(x):
    """Returns whether data or a Pending is considered "simple".

    "Simple" data is merged by identity, whereas this may not be the case
    for non-simple data, e.g. Possibilities are merged using set union, and
    distinct numbers e.g. 2 and 3 are merged into ANYTHING.

    Simple data can be forced more easily because it won't cause problems
    if we find more values to merge along.
    """
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
    """Represents pending data.

    Attributes:
        resolve: A function to call to resolve this Pending.
        priority: A nullary function that returns either None (if
            this Pending cannot be forced) or an integer. Pendings
            with higher priority will be merged first.
        loop: The InferenceLoop this Pending is attached to.
        equiv: A set of Pendings that this Pending is being merged
            with.

    """

    def __init__(self, resolve, priority, loop):
        """Initialize the Pending."""
        super().__init__(loop=loop)
        self.priority = priority
        if resolve is not None:
            self._resolve = resolve
        self.equiv = {self}

    def is_simple(self):
        """Return whether this Pending is simple or not.

        By default, this returns False.
        """
        return False

    def force_resolve(self):
        """Force a resolution."""
        self.set_result(self._resolve())

    def tie(self, other):
        """Tie to another Pending."""
        assert isinstance(other, Pending)
        self.equiv |= other.equiv
        e = self.equiv
        for p in self.equiv:
            p.equiv = e


class PendingFromList(Pending):
    """Represents a Pending that can take a value from a set.

    Attributes:
        possibilities: The set of values the Pending can take.
        default: The default value if we are forcing resolution.
        priority: A nullary function that returns either None (if
            this Pending cannot be forced) or an integer. Pendings
            with higher priority will be merged first.
        loop: The InferenceLoop this Pending is attached to.

    """

    def __init__(self, possibilities, default, priority, loop):
        """Initialize the PendingFromList."""
        super().__init__(
            resolve=None,
            priority=priority,
            loop=loop
        )
        self.default = default
        self.possibilities = possibilities

    def is_simple(self):
        """Returns whether all possibilities are simple."""
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
    """Represents a Pending with a tentative resolution.

    The tentative result may be updated until the Pending is done.
    """

    def __init__(self, tentative, loop):
        """Initialize the PendingTentative."""
        super().__init__(
            resolve=None,
            priority=self._priority,
            loop=loop
        )
        self.tentative = tentative

    def _priority(self):
        if isinstance(self.tentative, Pending):
            raise NotImplementedError()
        else:
            return -1001

    def force_resolve(self):
        """Resolve to the tentative value."""
        self.set_result(self.tentative)


async def find_coherent_result(v, fn):
    """Return fn(v) without fully resolving v, if possible.

    If v is a PendingFromList and fn(x) is the same for every x in v,
    this will return that result without resolving which possibility
    v is. Otherwise, v will be resolved.
    """
    if isinstance(v, PendingFromList):
        results = set()
        for option in v.possibilities:
            results.add(await fn(option))
        if len(results) == 1:
            return results.pop()
    x = await v
    return await fn(x)


async def force_pending(v):
    """Resolve v if v is Pending, otherwise return v directly."""
    if isinstance(v, Pending):
        return await v
    else:
        return v
