"""Core of the inference engine (not Myia-specific)."""

import asyncio
from contextvars import copy_context
from collections import deque

from ..dtype import Function, type_cloner_async, ismyiatype
from ..utils import Unification, Var, RestrictedVar, eprint, overload, \
    Overload

from .utils import InferenceError, DynamicMap, MyiaTypeError, ValueWrapper


class Later(Exception):
    pass


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

    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object(
            'Task',
            (('wait_for', self._fut_waiter),
             ('key', self.key)),
            delimiter="↦",
        )


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
        found = False
        later = []
        while self._vars:
            v1 = self._vars.pop()
            try:
                v1.force_resolve()
            except InferenceError as e:
                self._errors.append(e)
            except Later:
                later.append(v1)
            else:
                found = True
                break
        self._vars = later + self._vars
        return found

    def run_forever(self):
        """Run this loop until there is no more work to do."""
        while True:
            while self._todo:
                h = self._todo.popleft()
                h._run()
            self._vars = [fut for fut in self._vars if not fut.resolved()]
            if self._vars:
                # If some literals weren't forced to a concrete type by some
                # operation, we sort by priority (i.e. floats first) and we
                # force the first one to take its default concrete type. Then
                # we resume the loop.
                self._vars.sort(key=lambda x: x.priority)
                if self._resolve_var():
                    continue
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

    def create_future(self):
        """Create a Future using this loop."""
        return asyncio.Future(loop=self)

    def as_future(self, value):
        """Create a future that resolves to the given value."""
        if hasattr(value, '__await__'):
            return value
        else:
            fut = self.create_future()
            fut.set_result(value)
            return fut

    def create_var(self, var, default, priority=0):
        """Create an InferenceVar running on this loop."""
        v = InferenceVar(var, default, priority, loop=self)
        self._vars.append(v)
        return v

    def create_pending(self, resolve, priority=0, parents=()):
        pending = Pending(resolve, priority, loop=self)
        if priority is not None:
            self._vars.append(pending)
        return pending

    def create_pending_from_list(self, poss, dflt, priority=0):
        pending = PendingFromList(poss, dflt, priority, loop=self)
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


class EquivalenceChecker:
    """Handle equivalence between values."""

    def __init__(self, loop, error_callback):
        """Initialize the EquivalenceChecker."""
        self.loop = loop
        self.error_callback = error_callback

    def _make_eq(self, refs):
        def eq(x, y):
            """Equality check for x and y."""
            if isinstance(x, DynamicMap) and isinstance(y, DynamicMap):
                if x.provably_equivalent(y):
                    x.merge(y)
                    return True

                self._tie_dmaps(x, y, refs)
                self._tie_dmaps(y, x, refs)
                # We return True now, but if x and y are not equal, there
                # will be an error later.
                return True
            else:
                return x == y

        return eq

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

        if hasattr(x, '__await__') and not isinstance(x, InferenceVar):
            x = await x
        if hasattr(y, '__await__') and not isinstance(y, InferenceVar):
            y = await y

        if not self.merge(x, y, refs):
            error_callback(MyiaTypeMismatchError(x, y, refs=refs))

    def merge(self, x, y, refs=[]):
        """Merge the two values/variables x and y."""
        unif = Unification(eq=self._make_eq(refs))
        res = unif.unify(x, y, self.loop.equiv)
        if res is None:
            return False
        for var, value in res.items():
            # TODO: Only loop on unprocessed variables
            iv = var._infvar
            if isinstance(value, Var) and not hasattr(value, '_infvar'):
                # Unification may create additional variables
                self.loop.create_var(value, iv.default, iv.priority)
            if not iv.done():
                # NOTE: This updates self.loop.equiv
                iv.set_result(value)
        return True

    async def assert_same(self, *things, refs=[]):
        """Assert that all futures/values have the same value."""
        futs = [self.loop.as_future(x) for x in things]

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

        # Otherwise just return one of them
        return main.result()


def is_simple(x):
    if isinstance(x, Pending):
        return x.is_simple()
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
        if is_simple(self):
            for e in self.equiv:
                if isinstance(e, Pending) and not e.done():
                    e.set_result(value)

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


class InferenceVar(Pending):
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
        super().__init__(None, priority, loop=loop)
        self.loop = loop
        self.var = var
        self.default = default
        self.__var__ = var
        var._infvar = self

    def force_resolve(self):
        """Resolve to the default value."""
        default = self.default
        if default is None:
            if self.done():
                self.result()._infvar.force_resolve()
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

    def set_result(self, result):
        """Set the result of this InferenceVar.

        This updates the mapping in the loop's equivalence table.
        """
        self.loop.equiv[self.var] = result
        super().set_result(result)


async def find_coherent_result(infv, fn):
    """Try to apply fn on infv, resolving it only if needed.

    fn will be tried on every possible value infv can take, and if
    all results are the same, that result is returned. Otherwise,
    we await on infv to resolve it and we apply fn on that value.
    """
    v = infv.var
    if isinstance(v, RestrictedVar):
        results = set()
        for option in v.legal_values:
            results.add(await fn(option))
        if len(results) == 1:
            return results.pop()
    x = await infv
    return await fn(x)


async def find_coherent_result_2(v, fn):
    if isinstance(v, PendingFromList):
        results = set()
        for option in v.possibilities:
            results.add(await fn(option))
        if len(results) == 1:
            return results.pop()
    x = await v
    return await fn(x)


@overload(bootstrap=True)
async def reify_shallow(self, x: ValueWrapper):
    """Build a concrete value from v.

    Unlike reify which is deep, the outermost InferenceVar will be
    awaited on.
    """
    return await self(x.value)


@overload  # noqa: F811
async def reify_shallow(self, v: Var):
    return await self(v._infvar)


@overload  # noqa: F811
async def reify_shallow(self, v: Pending):
    return await self(await v)


# @overload  # noqa: F811
# async def reify_shallow(self, v: InferenceVar):
#     return await self(await v)


@overload  # noqa: F811
async def reify_shallow(self, v: object):
    return v


reify = Overload(mixins=[type_cloner_async, reify_shallow]).bootstrap()
