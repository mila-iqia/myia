"""Tools to handle contexts and references in inference."""

import asyncio
from dataclasses import dataclass
from .loop import force_pending


############
# Contexts #
############


class Context:
    """A context for the evaluation of a node.

    A context essentially contains the values of each relevant property of each
    parameter of each graph in which a node is nested.
    """

    @classmethod
    def empty(cls):
        """Return an empty context."""
        return _empty_context

    def __init__(self, parent, g, argkey):
        """Initialize the Context."""
        assert argkey is not None
        self.parent = parent
        self.graph = g
        self.argkey = argkey
        self.parent_cache = dict(parent.parent_cache) if parent else {}
        self.parent_cache[g] = self
        self._hash = hash((self.parent, self.graph, self.argkey))

    def filter(self, graph):
        """Return a context restricted to a graph's dependencies."""
        return self.parent_cache.get(graph, _empty_context)

    def add(self, graph, argkey):
        """Extend this context with values for another graph."""
        return Context(self.filter(graph.parent), graph, argkey)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return type(other) is Context \
            and self.parent == other.parent \
            and self.graph == other.graph \
            and self.argkey == other.argkey


_empty_context = Context(None, None, ())


class Contextless:
    """Singleton Context which specializes to itself.

    CONTEXTLESS is essentially an empty context that is idempotent under
    all operations. In practice it maps each node of each graph to a unique
    type, shape and so on.
    """

    @classmethod
    def empty(cls):
        """Return CONTEXTLESS."""
        return CONTEXTLESS

    def filter(self, graph):
        """Return CONTEXTLESS."""
        return self

    def add(self, graph, argvals):
        """Return CONTEXTLESS."""
        return self


CONTEXTLESS = Contextless()


##############
# References #
##############


class AbstractReference:
    """Superclass for Reference and VirtualReference."""


class Reference(AbstractReference):
    """Reference to a certain node in a certain context.

    Attributes:
        engine: The InferenceEngine in which this Reference lives.
        node: The ANFNode being referred to.
        context: The Context for the node.

    """

    def __init__(self, engine, node, context):
        """Initialize the Reference."""
        assert context is not None
        self.node = node
        self.engine = engine
        self.context = context
        self._hash = hash((self.node, self.context))

    async def get(self):
        """Get the value (asynchronous)."""
        raw = self.engine.get_inferred(self)
        return await force_pending(await raw)

    def get_sync(self):
        """Get the value (synchronous)."""
        return self.engine.run_coroutine(self.get(), throw=True)

    def get_resolved(self):
        """Get the value if resolved. Error out if not."""
        return self.engine.cache.cache[self].result()

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, Reference) \
            and self.node is other.node \
            and self.context == other.context


@dataclass
class VirtualReference(AbstractReference):
    """Synthetic reference that can be given to an inferrer.

    Attributes:
        abstract: The inferred value.

    """

    abstract: 'AbstractValue'  # noqa: F821

    async def get(self):
        """Get the value (asynchronous)."""
        return self.abstract

    def get_sync(self):  # pragma: no cover
        """Get the value (synchronous)."""
        return self.abstract


###################
# EvaluationCache #
###################


class EvaluationCache:
    """Key/value store where keys are associated to Futures.

    Attributes:
        cache: The cache.
        loop: The InferenceLoop for async evaluation.
        keycalc: An async function that takes a key and returns
            the value associated to that key.

    """

    def __init__(self, loop, keycalc, keytransform):
        """Initialize an EvaluationCache."""
        self.cache = {}
        self.loop = loop
        self.keytransform = keytransform
        self.keycalc = keycalc

    def get(self, key):
        """Get the future associated to the key."""
        key = self.keytransform(key)
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
