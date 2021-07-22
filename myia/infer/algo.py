"""Implementation of the inference algorithm."""

from collections import defaultdict
from types import GeneratorType

from ..abstract import data, utils as autils
from ..utils.misc import Named

NOT_DONE = Named("NOT_DONE")


class InferenceTask:
    """Represents an inference task.

    Arguments:
        node: The node being inferred.
        engine: The inference engine to use.

    Attributes:
        node: The node being inferred.
        inferrer: The (already started) generator handling this node.
        remap: A map from Generic/Placeholder instances to CanonGeneric
            instances.
        unif: The Unificator for this task.
    """

    def __init__(self, node, engine):
        self.node = node
        self.node.abstract = data.Placeholder()

        self.unif = autils.Unificator()

        # The inferrer is a generator and must receive the value None
        # at the beginning (because of Python's generator protocol)
        self.inferrer = engine(node, self.unif)
        self.remap = {}


class Scheduler:
    """Schedules the next step in the inference.

    Attributes:
        triggers: Associates each node to a list of pending InferenceRequests.
        satisfied: Associates each node to its inferred type.
        todo: List of (payload, output) tuples.
    """

    def __init__(self):
        self.triggers = defaultdict(list)
        self.satisfied = {None: None}
        self.todo = []

    def _satisfy_request(self, request, node, value):
        """Tell a request that the node's value has been inferred.

        If all the request's nodes have a value, the payload and
        request output will be appended to the todo list.
        """
        if request.satisfy(node, value):
            self.todo.append((request.payload, request.output))

    def require(self, request):
        """Register a request."""
        for node in request.nodes:
            if node in self.satisfied:
                self._satisfy_request(request, node, self.satisfied[node])
            else:
                self.triggers[node].append(request)

    def satisfy(self, node, value):
        """Associate an inferred value to a node and unblock satisfied requests."""
        self.satisfied[node] = value
        for request in self.triggers[node]:
            self._satisfy_request(request, node, value)
        self.triggers[node] = []

    def next(self):
        """Return the next thing to do.

        Returns (payload, output) corresponding to some request.
        """
        return self.todo.pop()


class InferenceRequest:
    """Request for the types of a list of nodes.

    Attributes:
        nodes: List of nodes to infer to satisfy this request.
        results: List of results corresponding to each node. Entries that
            are not yet available are the constant NOT_DONE.
        to_satisfy: Set of nodes for which there is no result as of yet.
        output: Request output, the result of self.combine(self.results).

    Set by the inferrer:
        unif: Unificator.
        payload: Data regarding the requester.
    """

    def __init__(self, *nodes):
        self.nodes = nodes
        self.unif = None
        self.payload = None
        self.results = [NOT_DONE for node in self.nodes]
        self.to_satisfy = set(self.nodes)
        self.output = NOT_DONE
        self.check()

    def check(self):
        """Check if the request is done."""
        done = not self.to_satisfy
        if done and self.output is NOT_DONE:
            # The request summarizes the results into the output
            # Different requests combine results differently
            self.output = self.combine(self.results)
        return done

    def satisfy(self, node, value):
        """Satisfy one of the requested nodes.

        Arguments:
            node: The node for which we have an inferred value.
            value: The inferred value for that node.

        Returns:
            Whether the task is done or not.
        """
        # The same node may be requested multiple times, so it will only
        # be in to_satisfy the first time around.
        if node in self.to_satisfy:
            self.to_satisfy.remove(node)
            # The results should be in the same order as the nodes
            for i, n in enumerate(self.nodes):
                if n is node:
                    self.results[i] = value
        return self.check()


class Require(InferenceRequest):
    """Request for a single node."""

    def combine(self, results):
        """Return the result."""
        (result,) = results
        return result


class RequireAll(InferenceRequest):
    """Request for a set of nodes."""

    def combine(self, results):
        """Return the results."""
        return tuple(results)


class Merge(InferenceRequest):
    """Request for a set of nodes, which must be merged.

    This differs from Unify on AbstractUnion, which are appended by
    Merge to form bigger unions, but unified by Unify.
    """

    def combine(self, results):
        """Merge the results."""
        x, y = results
        result, U = autils.merge(x, y, U=self.unif)
        return result


class Unify(InferenceRequest):
    """Request for a set of nodes, which must be unified.

    This differs from Merge on AbstractUnion, which are appended by
    Merge to form bigger unions, but unified by Unify.
    """

    def combine(self, results):
        """Unify the results."""
        x, y = results
        result, U = autils.unify(x, y, U=self.unif)
        return result


class Inferrer:
    """Inference algorithm.

    Arguments:
        engine: A generator that takes a node to infer and yields requests for
            other nodes, to drive the inference.

    Attributes:
        unif: A Unificator that resolves generics and placeholders.
        scheduler: Schedules the next node to infer.
    """

    def __init__(self, engine):
        self.engine = engine
        self.unif = autils.Unificator()
        self.scheduler = Scheduler()

    def bootstrap(self, node):
        """Start inference of the node.

        If the node was not already inferred, set the inference result to a
        placeholder and call `self.engine(node)` to start the inference.
        """
        existing = node.abstract
        if existing is None:
            self.step(
                task=InferenceTask(node=node, engine=self.engine),
                value=None,
            )
        else:
            self.scheduler.satisfy(node, existing)

    def step(self, task, value):
        """Perform an inference step.

        Arguments:
            task: The InferenceTask.
            value: The value to give to the inferrer to continue.
        """
        node = task.node
        inferrer = task.inferrer

        if inferrer.gi_frame is None:
            # This inferrer has ended.
            return

        assert isinstance(inferrer, GeneratorType)
        try:
            # Basically performs a renaming on all type variables in value so that
            # the inferrer only sees CanonGeneric. Basically, whether the value is
            # (placeholder103, placeholder157), or (placeholder115, placeholder204),
            # the inferrer should behave the same. Therefore, we map both of them
            # to (canon1, canon2) so that the inferrer can easily cache results.
            # We will later reverse the mapping to unify on the real placeholders.
            if isinstance(value, tuple):
                value = tuple(
                    autils.canonical(x, mapping=task.remap) for x in value
                )
            else:
                value = autils.canonical(value, mapping=task.remap)

            # If the inferrer is not done it will yield an additional request
            request = inferrer.send(value)

            # Fill in extra attributes in the request
            request.unif = self.unif
            request.payload = task

            # All requested nodes are scheduled
            for rnode in request.nodes:
                self.bootstrap(rnode)

            # Schedule the rest of the inferrer, which will be triggered when all
            # nodes in the request are resolved.
            self.scheduler.require(request)

        except StopIteration as stop:
            # The inferrer returns the type for the node
            assert stop.value is not None
            # We need to undo the renaming we performed in order to recover the
            # actual placeholders.
            result = autils.uncanonical(stop.value, mapping=task.remap)

            # Unify with the local unificator
            for k, v in task.remap.items():
                autils.unify(k, v, U=task.unif)
            autils.unify(result, node.abstract, U=task.unif)

            # We then unify the current node's placeholder (in node.abstract)
            # with the result and set node.abstract to that.
            node.abstract, U = autils.unify(result, node.abstract, U=self.unif)

            # Declare the node's type
            self.scheduler.satisfy(node, node.abstract)

    def run(self, start):
        """Run inference, starting from the start node."""
        self.bootstrap(start)
        while True:
            try:
                task, value = self.scheduler.next()
            except IndexError:
                # Nothing more to do
                break

            self.step(task, value)

        # The result for start will be in start.abstract
        result = self.unif.canon.get(start.abstract, start.abstract)
        start.abstract = result
        assert not isinstance(result, data.Placeholder)
        return result


def infer(engine, node):
    """Infer the type of node using an inference engine.

    Arguments:
        engine: The inference engine, a generator function that takes a node
            as an argument, yields InferenceRequests as needed, and returns
            the type for the node.
        node: The node to infer.
    """
    inf = Inferrer(engine)
    return inf.run(node)
