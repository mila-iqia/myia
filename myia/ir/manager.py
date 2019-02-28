"""Managing graph modification and information about graphs."""


from collections import defaultdict, Counter

from ..graph_utils import dfs, FOLLOW, EXCLUDE
from ..utils import Events, Partializable, OrderedSet

from .utils import succ_deeper


class ManagerError(Exception):
    """Class for errors raised by GraphManager."""


def manage(*graphs, weak=False):
    """Ensure that all given graphs have a manager and return it.

    * If one or more graphs has a manager, that manager will be used.
    * If two graphs have different managers, an error will be raised.
    * If no graph has a manager, one will be created.

    Args:
        graphs: The graphs to manage.
        weak: If True, when creating a new manager, graphs will not
            be forcefully associated with it. (Defaults to False.)
    """
    manager = None
    for graph in graphs:
        manager = graph._manager
        if manager:
            break
    if manager is None:
        manager = GraphManager(manage=not weak)
        root = True
    else:
        root = False
    for graph in graphs:
        manager.add_graph(graph, root=root)
    return manager


class ParentProxy:
    """Represents a graph's immediate parent."""

    def __init__(self, graph):
        """Initialize the ParentProxy."""
        self.graph = graph

    def __hash__(self):
        return hash(self.graph)

    def __eq__(self, other):
        return isinstance(other, ParentProxy) and other.graph is self.graph


class PerGraphStatistic(dict):
    """Represents a statistic that maps each graph to some information."""

    constructor = dict
    include_graph_none = False

    def __init__(self, manager):
        """Initialize a PerGraphStatistic."""
        super().__init__()
        self.manager = manager
        evts = self.manager.events
        evts.add_node.register(self._on_add_node)
        evts.drop_node.register(self._on_drop_node)
        evts.add_graph.register(self._on_add_graph)
        evts.drop_graph.register(self._on_drop_graph)
        evts.add_edge.register(self._on_add_edge)
        evts.drop_edge.register(self._on_drop_edge)
        if self.include_graph_none:
            self[None] = self.constructor()

    def reset(self):
        """Reset this graph's information."""
        return self.clear()

    def _on_add_graph(self, event, graph):
        self[graph] = self.constructor()

    def _on_drop_graph(self, event, graph):
        del self[graph]

    def _on_add_node(self, event, node):
        pass

    def _on_drop_node(self, event, node):
        pass

    def _on_add_edge(self, event, node, key, value):
        pass

    def _on_drop_edge(self, event, node, key, value):
        pass


class NodesStatistic(PerGraphStatistic):
    """Implements `GraphManager.nodes`."""

    constructor = OrderedSet
    include_graph_none = True

    def _on_add_node(self, event, node):
        self[node.graph].add(node)

    def _on_drop_node(self, event, node):
        self[node.graph].remove(node)


class CounterStatistic(PerGraphStatistic):
    """Represents a statistic that maps each graph to a set of counters."""

    def inc(self, graph, key, qty=1):
        """Increment the count for self[graph][key] by qty."""
        d = self[graph]
        if key not in d:
            d[key] = qty
            return True
        else:
            d[key] += qty
            return False

    def dec(self, graph, key, qty=1):
        """Decrement the count for self[graph][key] by qty.

        The key is deleted if the count falls to zero.
        """
        d = self[graph]
        if d[key] == qty:
            del d[key]
            return True
        else:
            d[key] -= qty
            assert d[key] > 0
            return False

    def mod(self, graph, key, qty):
        """Change the count for self[graph][key] by qty."""
        if qty > 0:
            return self.inc(graph, key, qty)
        elif qty < 0:
            return self.dec(graph, key, -qty)
        else:
            # If this happens, there's probably a bug elsewhere,
            # or a new features, in which case this can be a no-op
            raise ValueError('qty cannot be 0')  # pragma: no cover

    def _on_mod_edge(self, event, node, key, inp, direction):
        raise NotImplementedError()

    def _on_add_edge(self, event, node, key, inp):
        self._on_mod_edge(event, node, key, inp, 1)

    def _on_drop_edge(self, event, node, key, inp):
        self._on_mod_edge(event, node, key, inp, -1)


class ConstantsStatistic(CounterStatistic):
    """Implements `GraphManager.constants`."""

    def _on_mod_edge(self, event, node, key, inp, direction):
        if inp.is_constant():
            self.mod(node.graph, inp, direction)


class GraphConstantsStatistic(CounterStatistic):
    """Implements `GraphManager.graph_constants`."""

    def _on_mod_edge(self, event, node, key, inp, direction):
        if inp.is_constant_graph():
            self.mod(inp.value, inp, direction)


class FVDirectStatistic(CounterStatistic):
    """Implements `GraphManager.free_variables_direct`."""

    def _on_mod_edge(self, event, node, key, inp, direction):
        g1 = node.graph
        g2 = inp.graph
        if g1 and g2 and g1 is not g2:
            self.mod(g1, inp, direction)


class GDepDirectStatistic(CounterStatistic):
    """Implements `GraphManager.graph_dependencies_direct`."""

    def _on_mod_edge(self, event, node, key, inp, direction):
        g1 = node.graph
        g2 = inp.graph
        if g1 and g2 and g1 is not g2:
            self.mod(g1, g2, direction)


class GDepProxStatistic(CounterStatistic):
    """Implements `GraphManager.graph_dependencies_prox`."""

    def _on_mod_edge(self, event, node, key, inp, direction):
        g1 = node.graph

        if inp.is_constant_graph():
            ig = inp.value
            if self.mod(g1, ParentProxy(ig), direction):
                self.manager.events.invalidate_nesting()
        else:
            g2 = inp.graph
            if g1 and g2 and g1 is not g2:
                if self.mod(g1, g2, direction):
                    self.manager.events.invalidate_nesting()


class GDepProxInvStatistic(CounterStatistic):
    """Implements `GraphManager.graph_dependencies_prox_inv`."""

    def _on_mod_edge(self, event, node, key, inp, direction):
        g1 = node.graph

        if inp.is_constant_graph():
            ig = inp.value
            self.mod(ig, ParentProxy(g1), direction)
        else:
            g2 = inp.graph
            if g1 and g2 and g1 is not g2:
                self.mod(g2, g1, direction)


class GraphsUsedStatistic(CounterStatistic):
    """Implements `GraphManager.graphs_used`."""

    def _on_mod_edge(self, event, node, key, inp, direction):
        if inp.is_constant_graph():
            if self.mod(node.graph, inp.value, direction):
                self.manager.events.invalidate_uses()


class GraphUsersStatistic(CounterStatistic):
    """Implements `GraphManager.graph_users`."""

    def _on_mod_edge(self, event, node, key, inp, direction):
        if inp.is_constant_graph():
            self.mod(inp.value, node.graph, direction)


class NestingStatistic(PerGraphStatistic):
    """Represents a statistic about nesting.

    These statistics become invalid when the `invalidate_nesting` event
    is fired.
    """

    def __init__(self, manager):
        """Initialize a NestingStatistic."""
        super().__init__(manager)
        evts = self.manager.events
        evts.invalidate_nesting.register(self._on_invalidate_nesting)
        self.valid = False

    def _on_invalidate_nesting(self, event):
        self.reset()

    def _on_add_graph(self, event, graph):
        self.reset()

    def _on_drop_graph(self, event, graph):
        self.reset()


class NestingStatisticWholesale(NestingStatistic):
    """Statistic about nesting, computed for all graphs at once."""

    def __init__(self, manager):
        """Initialize a NestingStatistic."""
        super().__init__(manager)
        self.valid = False

    def reset(self):
        """Reset this graph's information.

        This makes the statistic invalid, so it must be recomputed.
        """
        super().reset()
        self.valid = False

    def recompute(self):
        """Recompute the information from scratch."""
        self._recompute()
        self.valid = True

    def _recompute(self):
        raise NotImplementedError()


class NestingStatisticGraphWise(NestingStatistic):
    """Statistic about nesting, computed on demand."""

    def __getitem__(self, g):
        if g in self:
            return super().__getitem__(g)
        else:
            res = self._compute(g)
            self[g] = res
            return res


class UsesStatistic(PerGraphStatistic):
    """Represents a statistic about uses.

    These statistics become invalid when the `invalidate_uses` event
    is fired.
    """

    def __init__(self, manager):
        """Initialize a NestingStatistic."""
        super().__init__(manager)
        evts = self.manager.events
        evts.invalidate_uses.register(self._on_invalidate_uses)

    def _on_invalidate_uses(self, event):
        self.reset()

    def _on_add_graph(self, event, graph):
        self.reset()

    def _on_drop_graph(self, event, graph):
        self.reset()

    def __getitem__(self, g):
        if g in self:
            return super().__getitem__(g)
        else:
            res = self._compute(g)
            self[g] = res
            return res


class GDepTotalStatistic(NestingStatisticGraphWise):
    """Implements `GraphManager.graph_dependencies_total`."""

    def _compute(self, g, path=None):
        if g in self:
            return self.get(g)
        if path is None:
            path = set()
        if g in path:
            return set()
        all_deps = self.manager.graph_dependencies_prox
        deps = all_deps[g]
        parents = set()
        for dep in deps:
            if isinstance(dep, ParentProxy):
                parents |= self._compute(dep.graph, path | {g})
            else:
                parents.add(dep)
        parents.discard(g)
        self[g] = parents
        return parents


class ParentStatistic(NestingStatisticGraphWise):
    """Implements `GraphManager.parents`."""

    def _compute(self, g):
        if g in self:
            return self.get(g)
        all_deps = self.manager.graph_dependencies_total
        todo = set(all_deps[g])
        deps = set(all_deps[g])
        while todo:
            # We eliminate all grandparents
            g2 = todo.pop()
            p = self._compute(g2)
            while p is not None:
                deps.discard(p)
                todo.discard(p)
                p = self.get(p)
        if len(deps) == 0:
            self[g] = None
        elif len(deps) == 1:
            self[g], = deps
        else:
            # The way code is structured, this should never happen
            raise AssertionError('Too many parents')
        return self.get(g)


class ChildrenStatistic(NestingStatisticGraphWise):
    """Implements `GraphManager.children`."""

    def _compute(self, g):
        parents = self.manager.parents
        reach = self.manager.graphs_reachable
        return {g2 for g2 in reach[g] if parents[g2] is g}


class ScopeStatistic(NestingStatisticGraphWise):
    """Implements `GraphManager.scopes`."""

    def _compute(self, g):
        children = self.manager.children
        scope = {g}
        for child in children[g]:
            scope.update(self._compute(child))
        return scope


class FVTotalStatistic(NestingStatisticWholesale, CounterStatistic):
    """Implements `GraphManager.free_variables_total`."""

    def _recompute(self):
        mng = self.manager

        for g in mng.graphs:
            self[g] = {}

        for g in mng.graphs:
            for node, count in mng.free_variables_direct[g].items():
                curr = g
                while curr:
                    self.mod(curr, node, count)
                    curr = mng.parents[curr]
                    if node in mng.nodes[curr]:
                        break

            for g2, count in mng.graphs_used[g].items():
                p = mng.parents[g2]
                if p is None:
                    continue
                curr = g
                while curr is not p:
                    self.mod(curr, g2, count)
                    curr = mng.parents[curr]

    def _on_mod_edge(self, event, node, key, inp, direction):
        if not self.valid:
            return

        g1 = node.graph

        def _update(stop_graph, fv):
            curr = g1
            while curr and curr is not stop_graph:
                self.mod(curr, fv, direction)
                curr = self.manager.parents[curr]

        if inp.is_constant_graph():
            ig = inp.value
            p = self.manager.parents[ig]
            if p:
                _update(p, ig)

        g2 = inp.graph
        if g1 and g2 and g1 is not g2:
            _update(g2, inp)


class GraphsReachableStatistic(UsesStatistic):
    """Implements `GraphManager.graphs_reachable`."""

    def _compute(self, g):
        used = self.manager.graphs_used
        todo = {g}
        seen = set()
        reach = set()
        while todo:
            g2 = todo.pop()
            if g2 in seen:
                continue
            todo.update(used[g2])
            reach.update(used[g2])
            seen.add(g2)
        return reach


class RecursiveStatistic(UsesStatistic):
    """Implements `GraphManager.recursive`."""

    def __getitem__(self, g):
        reach = self.manager.graphs_reachable
        return g in reach[g]


class GraphManager(Partializable):
    """Structure to hold information about graphs and modify them.

    Attributes are updated incrementally when graph mutations are committed.

    Properties are updated incrementally when possible, but may be invalidated
    when graph dependencies change. In that case they will be recomputed lazily
    the next time they are requested.

    Attributes:
        all_nodes:
            Set of all nodes in all graphs managed by this GraphManager.

        nodes:
            Map each graph to the set of nodes that belong to it.

        uses:
            Map each node to the set of nodes that point to it.

        free_variables_direct:
            Map each graph to its free variables.

            "Direct" free variables are those that the graph refers
            to directly. Nested graphs are not taken into account, but
            they are in `free_variables_total`.

        graphs_used:
            Map each graph to the set of graphs it uses. For each graph,
            this is the set of graphs that it refers to directly.

        graph_users:
            Map each graph to the set of graphs that use it. For each graph,
            this is the set of graphs that refer to it directly.

        graph_dependencies_direct:
            Map each graph to the graphs it gets free variables from.

            This is the set of graphs that own the nodes returned by
            `free_variables_direct`, for each graph.

        graph_constants:
            Map each graph to the set of Constant nodes that have this
            graph as a value.

        graph_dependencies_total:
            Map each graph to the set of graphs it depends on. This is a
            superset of `graph_dependencies_direct` which also includes the
            graphs from which nested graphs need free variables.

        parents:
            Map each graph to its parent graph. Top-level graphs are
            associated to `None` in the returned dictionary.

        children:
            Map each graph to the graphs immediately nested in it. This is the
            inverse map of `parents`.

        scopes:
            Map each graph to the complete set of graphs nested in it. The set
            associated to a graph includes the graph itself.

        graphs_reachable:
            Map each graph to the set of graphs that may be called from there.
            For a given graph, this is the set of graphs it directly refers
            to plus the set of graphs that these graphs refer to, and so on.

        recursive:
            Map each graph to whether it is recursive of not. This amounts to
            whether the graph is reachable from itself, so if f calls g and
            g calls f, both f and g are recursive.

    """

    def __init__(self, *roots, manage=True):
        """Initialize the GraphManager."""
        self.roots = roots
        self.manage = manage
        self.reset()

    def reset(self):
        """Reset the manager's state.

        Recompute everything from the roots.
        """
        self.events = Events(
            add_node=None,
            drop_node=None,
            add_graph=None,
            drop_graph=None,
            add_edge=None,
            drop_edge=None,
            invalidate_nesting=None,
            invalidate_uses=None,
        )
        roots = OrderedSet(self.roots) if self.roots else OrderedSet()
        self.roots = OrderedSet()
        self.graphs = OrderedSet()
        self.all_nodes = OrderedSet()
        self.uses = defaultdict(OrderedSet)

        self.nodes = NodesStatistic(self)
        self.constants = ConstantsStatistic(self)
        self.free_variables_direct = FVDirectStatistic(self)
        self.graph_constants = GraphConstantsStatistic(self)
        self.graphs_used = GraphsUsedStatistic(self)
        self.graph_users = GraphUsersStatistic(self)
        self.graph_dependencies_direct = GDepDirectStatistic(self)
        self.graph_dependencies_prox = GDepProxStatistic(self)
        self.graph_dependencies_prox_inv = GDepProxInvStatistic(self)

        self.graph_dependencies_total = GDepTotalStatistic(self)
        self.parents = ParentStatistic(self)
        self.children = ChildrenStatistic(self)
        self.scopes = ScopeStatistic(self)
        self._free_variables_total = FVTotalStatistic(self)
        self.graphs_reachable = GraphsReachableStatistic(self)
        self.recursive = RecursiveStatistic(self)

        for root in roots:
            self.add_graph(root, root=True)

    def add_graph(self, graph, root=False):
        """Add a graph to this manager, optionally as a root graph."""
        if root:
            self.roots.add(graph)
        if graph in self.graphs:
            return
        self._ensure_graph(graph)
        self.events.add_graph(graph)
        self._acquire_nodes(graph.parameters)
        self._acquire_nodes({graph.return_})

    def keep_roots(self, *roots):
        """Keep only the graphs reachable from the given roots.

        All other graphs will be removed from this manager.

        If no roots are given, existing roots will be used.
        """
        if roots:
            self.roots = OrderedSet()
            for root in roots:
                self.add_graph(root, True)
        else:
            roots = self.roots
        keep = OrderedSet()
        for root in roots:
            keep.update(self.graphs_reachable[root])
        self._maybe_drop_graphs(self.graphs - keep, ignore_users=True)

    def _ensure_graph(self, graph):
        """Ensure that the graph is managed by this manager."""
        if self.manage:
            if graph._manager and graph._manager is not self:
                raise ManagerError('A graph can only have one manager.')
            graph._manager = self
        self.graphs.add(graph)

    def _maybe_drop_graphs(self, graphs, ignore_users=False):
        todo = OrderedSet(graphs)
        dropped = set()

        while todo:
            graph = todo.pop()

            if graph in self.roots:
                continue

            users = self.graph_users[graph]
            if users and not ignore_users:
                continue

            dropped.add(graph)

            todo |= self._maybe_drop_nodes(OrderedSet([graph.return_]))

        for g in dropped:
            self.events.drop_graph(g)
            self.all_nodes.difference_update(g.parameters)
            self.graphs.remove(g)
            if g._manager is self:
                g._manager = None

    def _process_edge(self, node, key, inp, direction):
        """Add/remove an edge between two nodes.

        Args:
            direction:
                * 1 if the edge is added.
                * -1 if the edge is removed.
        """
        if direction == -1:
            if (node, key) not in self.uses[inp]:
                # It's possible that we already got here when we
                # dropped a graph.
                return  # pragma: no cover
            self.uses[inp].remove((node, key))
            self.events.drop_edge(node, key, inp)
        else:
            if inp.graph is not None:
                self.add_graph(inp.graph)
            if inp.is_constant_graph():
                self.add_graph(inp.value)
            self.uses[inp].add((node, key))
            self.events.add_edge(node, key, inp)

    def _process_inputs(self, node, direction):
        """Process the inputs of a newly [dis]connected node.

        Args:
            direction:
                * 1 if the node was connected.
                * -1 if the node was disconnected.
        """
        for key, inp in enumerate(node.inputs):
            self._process_edge(node, key, inp, direction)

    def _acquire_nodes(self, nodes):
        """Add newly connected nodes."""
        def limit(x):
            if x in self.all_nodes:
                return EXCLUDE
            else:
                return FOLLOW

        acq = OrderedSet()
        for node in nodes:
            new_nodes = OrderedSet(dfs(node, succ_deeper, limit))
            self.all_nodes |= new_nodes
            acq |= new_nodes

        for node in acq:
            g = node.graph
            if g is not None:
                self.add_graph(g)
            self.events.add_node(node)
            self._process_inputs(node, 1)

    def _maybe_drop_nodes(self, nodes):
        """Check if the nodes are connected to a graph, drop them if not."""
        nodes = OrderedSet(nodes)

        # Set of graphs to check if we want to drop them or not
        graphs_to_check = OrderedSet()

        while nodes:
            node = nodes.pop()
            if node not in self.all_nodes:
                continue
            uses = self.uses[node]

            if uses or (node.is_parameter() and node in node.graph.parameters):
                continue

            if node.is_constant_graph():
                graphs_to_check.add(node.value)

            self._process_inputs(node, -1)
            self.all_nodes.remove(node)
            self.events.drop_node(node)
            nodes.update(node.inputs)

        return graphs_to_check

    def _ensure_statistic(self, stat):
        if not stat.valid:
            stat.recompute()
        return stat

    @property
    def free_variables_total(self):
        """Map each graph to its free variables.

        This differs from `free_variables_direct` in that it also includes free
        variables needed by children graphs. Furthermore, graph Constants may
        figure as free variables.
        """
        return self._ensure_statistic(self._free_variables_total)

    def set_parameters(self, graph, parameters):
        """Replace a graph's parameters."""
        with self.transact() as tr:
            tr.set_parameters(graph, parameters)

    def replace(self, old_node, new_node):
        """Replace old_node by new_node."""
        with self.transact() as tr:
            tr.replace(old_node, new_node)

    def set_edge(self, node, key, value):
        """Set node.inputs[key] to value."""
        with self.transact() as tr:
            tr.set_edge(node, key, value)

    def transact(self):
        """Begin a transaction.

        >>> with mng.transact() as tr:
        ...     tr.replace(node1, node2)
        ...     ...
        """
        return GraphTransaction(self)

    def _commit_changes(self, changes):
        """Commit changes.

        This modifies the graph and update attributes and properties.
        """
        addedges = Counter()
        rmedges = Counter()

        adds = Counter()
        rms = Counter()

        for operation, *args in changes:
            if operation == 'set_edge':
                root_node, key, new_node = args
                old_node = root_node.inputs[key]
                rmedges[(root_node, key, old_node)] += 1
                addedges[(root_node, key, new_node)] += 1
                rms[old_node] += 1
                adds[new_node] += 1
                root_node.inputs[key] = new_node
            elif operation == 'set_parameters':
                graph, new_parameters = args
                old_parameters = graph.parameters
                for p in new_parameters:
                    adds[p] += 1
                for p in old_parameters:
                    rms[p] += 1
                graph.parameters = new_parameters

        for root_node, key, new_node in addedges - rmedges:
            self._process_edge(root_node, key, new_node, 1)

        self._acquire_nodes(adds - rms)

        for root_node, key, old_node in rmedges - addedges:
            self._process_edge(root_node, key, old_node, -1)

        maybe_drop_graphs = self._maybe_drop_nodes(rms - adds)

        self._maybe_drop_graphs(maybe_drop_graphs)


class GraphTransaction:
    """Group changes to a graph into a transaction.

    GraphTransaction supports replacing nodes, setting edges and replacing a
    graph's parameters list. No changes actually happen until the commit()
    method is called. commit() can only be called once, for multiple
    transactions create multiple GraphTransaction objects.

    When used as a context manager, commit() is called automatically at the end
    of the with block.
    """

    def __init__(self, manager):
        """Initialize a GraphTransaction."""
        if not manager.manage:
            raise ManagerError('Cannot modify graph through this manager')
        self.manager = manager
        self.changes = []

    def set_parameters(self, graph, parameters):
        """Declare replacement of a graph's parameters."""
        self.changes.append(('set_parameters', graph, parameters))

    def replace(self, old_node, new_node):
        """Declare replacement of old_node by new_node.

        This does not modify the graph. The change will occur in the next call
        to commit().
        """
        g = old_node.graph
        if g and g.return_ is old_node:
            raise ManagerError('Cannot replace the return node of a graph.')
        uses = self.manager.uses[old_node]
        for node, key in uses:
            self.set_edge(node, key, new_node)

    def set_edge(self, node, key, value):
        """Declare setting node.inputs[key] to value.

        This does not modify the graph. The change will occur in the next call
        to commit().
        """
        self.changes.append(('set_edge', node, key, value))

    def commit(self):
        """Commit the changes."""
        changes, self.changes = self.changes, None
        self.manager._commit_changes(changes)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        if type is None:
            self.commit()
