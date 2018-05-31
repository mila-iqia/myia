"""Managing graph modification and information about graphs."""


from collections import defaultdict, Counter

from .ir import succ_deeper, is_constant_graph
from .graph_utils import dfs, FOLLOW, EXCLUDE
from .utils import Events


def manage(*graphs):
    """Ensure that all given graphs have a manager and return it.

    * If one or more graphs has a manager, that manager will be used.
    * If two graphs have different managers, an error will be raised.
    * If no graph has a manager, one will be created.
    """
    manager = None
    for graph in graphs:
        manager = graph._manager
    if manager is None:
        manager = GraphManager()
    for graph in graphs:
        manager.add_graph(graph, root=True)
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

    constructor = set
    include_graph_none = True

    def _on_add_node(self, event, node):
        self[node.graph].add(node)

    def _on_drop_node(self, event, node):
        self[node.graph].remove(node)


class CounterStatistic(PerGraphStatistic):
    """Represents a statistic that maps each graph to a set of counters."""

    constructor = dict

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
            return False

    def mod(self, graph, key, qty):
        """Change the count for self[graph][key] by qty."""
        if qty > 0:
            return self.inc(graph, key, qty)
        elif qty < 0:
            return self.dec(graph, key, -qty)
        else:
            raise ValueError('qty cannot be 0')  # pragma: no cover

    def _on_add_edge(self, event, node, key, inp):
        self._on_mod_edge(event, node, key, inp, 1)

    def _on_drop_edge(self, event, node, key, inp):
        self._on_mod_edge(event, node, key, inp, -1)


class ConstantsStatistic(CounterStatistic):
    """Implements `GraphManager.constants`."""

    def _on_mod_edge(self, event, node, key, inp, direction):
        if is_constant(inp):
            self.mod(node.graph, inp, direction)


class GraphConstantsStatistic(CounterStatistic):
    """Implements `GraphManager.graph_constants`."""

    def _on_mod_edge(self, event, node, key, inp, direction):
        if is_constant_graph(inp):
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

        if is_constant_graph(inp):
            ig = inp.value
            if self.mod(g1, ParentProxy(ig), direction):
                self.manager.events.invalidate_nesting()

        g2 = inp.graph
        if g1 and g2 and g1 is not g2:
            if self.mod(g1, g2, direction):
                self.manager.events.invalidate_nesting()


class GraphsUsedStatistic(CounterStatistic):
    """Implements `GraphManager.graphs_used`."""

    def _on_mod_edge(self, event, node, key, inp, direction):
        if is_constant_graph(inp):
            self.mod(node.graph, inp.value, direction)


class GraphUsersStatistic(CounterStatistic):
    """Implements `GraphManager.graph_users`."""

    def _on_mod_edge(self, event, node, key, inp, direction):
        if is_constant_graph(inp):
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

    def reset(self):
        """Reset this graph's information.

        This makes the statistic invalid, so it must be recomputed.
        """
        super().reset()
        self.valid = False

    def _on_invalidate_nesting(self, event):
        self.reset()

    def _on_add_graph(self, event, graph):
        self.reset()

    def _on_drop_graph(self, event, graph):
        self.reset()

    def recompute(self):
        """Recompute the information from scratch."""
        self._recompute()
        self.valid = True

    def _recompute(self):
        raise NotImplementedError()


class GDepTotalStatistic(NestingStatistic):
    """Implements `GraphManager.graph_dependencies_total`."""

    def _recompute(self):
        all_deps = self.manager.graph_dependencies_prox

        def seek_parents(g, path=None):
            if path is None:
                path = set()
            if g in path:
                return set()
            deps = all_deps[g]
            parents = set()
            for dep in deps:
                if isinstance(dep, ParentProxy):
                    parents |= seek_parents(dep.graph, path | {g})
                else:
                    parents.add(dep)
            return parents - {g}

        for g in list(all_deps.keys()):
            self[g] = seek_parents(g)


class ParentStatistic(NestingStatistic):
    """Implements `GraphManager.parents`."""

    def _recompute(self):
        for g in self.manager.graphs:
            self[g] = None

        all_deps = self.manager.graph_dependencies_total
        todo = [(g, set(deps)) for g, deps in all_deps.items()]
        todo.sort(key=lambda xy: len(xy[1]))

        while todo:
            next_todo = []
            for g, deps in todo:
                if len(deps) > 1:
                    rm = set()
                    for dep in deps:
                        parent = self[dep]
                        while parent:
                            rm.add(parent)
                            parent = self[parent]
                    deps -= rm
                if len(deps) == 0:
                    self[g] = None
                elif len(deps) == 1:
                    self[g] = deps.pop()
                else:
                    next_todo.append((g, deps))
            todo = next_todo


class ChildrenStatistic(NestingStatistic):
    """Implements `GraphManager.children`."""

    def _recompute(self):
        parents = self.manager.parents
        for g in self.manager.graphs:
            self[g] = set()
        for g, parent in parents.items():
            if parent is not None:
                self[parent].add(g)


class ScopeStatistic(NestingStatistic):
    """Implements `GraphManager.scopes`."""

    def _recompute(self):
        parents = self.manager.parents
        for g in self.manager.graphs:
            self[g] = set()
        for g in self.manager.graphs:
            p = g
            while p:
                self[p].add(g)
                p = parents[p]


class FVTotalStatistic(NestingStatistic, CounterStatistic):
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

        if is_constant_graph(inp):
            ig = inp.value
            if self.manager._parents.valid:
                p = self.manager._parents[ig]
                if p:
                    _update(p, ig)

        g2 = inp.graph
        if g1 and g2 and g1 is not g2:
            _update(g2, inp)


class GraphManager:
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

    """

    def __init__(self, *roots):
        """Initialize the GraphManager."""
        self._changes = []
        self.roots = roots
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
        )
        roots = set(self.roots) if self.roots else set()
        self.roots = set()
        self.graphs = set()
        self.all_nodes = set()
        self.uses = defaultdict(set)

        self.nodes = NodesStatistic(self)
        self.constants = ConstantsStatistic(self)
        self.free_variables_direct = FVDirectStatistic(self)
        self.graph_constants = GraphConstantsStatistic(self)
        self.graphs_used = GraphsUsedStatistic(self)
        self.graph_users = GraphUsersStatistic(self)
        self.graph_dependencies_direct = GDepDirectStatistic(self)
        self.graph_dependencies_prox = GDepProxStatistic(self)

        self._graph_dependencies_total = GDepTotalStatistic(self)
        self._parents = ParentStatistic(self)
        self._children = ChildrenStatistic(self)
        self._scopes = ScopeStatistic(self)
        self._free_variables_total = FVTotalStatistic(self)

        for root in roots:
            self.add_graph(root, root=True)

    def add_graph(self, graph, root=False):
        """Add a graph to this manager, optionally as a root graph."""
        if graph in self.graphs:
            return
        self._ensure_graph(graph)
        self.events.add_graph(graph)
        if root:
            self.roots.add(graph)
        self._acquire_nodes({graph.return_})

    def _ensure_graph(self, graph):
        """Ensure that the graph is managed by this manager."""
        if graph._manager and graph._manager is not self:
            raise Exception('A graph can only have one manager.')
        graph._manager = self
        self.graphs.add(graph)

    def _process_edge(self, node, key, inp, direction):
        """Add/remove an edge between two nodes.

        Args:
            direction:
                * 1 if the edge is added.
                * -1 if the edge is removed.
        """
        if inp.graph is not None:
            self.add_graph(inp.graph)
        if is_constant_graph(inp):
            self.add_graph(inp.value)

        if direction == -1:
            self.uses[inp].remove((node, key))
            self.events.drop_edge(node, key, inp)
        else:
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

        acq = set()
        for node in nodes:
            new_nodes = set(dfs(node, succ_deeper, limit))
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
        nodes = set(nodes)

        while nodes:
            node = nodes.pop()
            assert node in self.all_nodes
            uses = self.uses[node]
            if uses or (node.graph
                        and node.graph in self.graphs
                        and node is node.graph.return_):
                # This node is still live
                continue  # pragma: no cover

            self.all_nodes.remove(node)
            self.events.drop_node(node)
            self._process_inputs(node, -1)
            nodes.update(node.inputs)

    def _ensure_statistic(self, stat):
        if not stat.valid:
            stat.recompute()
        return stat

    @property
    def graph_dependencies_total(self):
        """Map each graph to the set of graphs it depends on.

        This is a superset of `graph_dependencies_direct` which also
        includes the graphs from which nested graphs need free
        variables.
        """
        return self._ensure_statistic(self._graph_dependencies_total)

    @property
    def parents(self):
        """Map each graph to its parent graph.

        Top-level graphs are associated to `None` in the returned
        dictionary.
        """
        return self._ensure_statistic(self._parents)

    @property
    def children(self):
        """Map each graph to the graphs immediately nested in it.

        This is the inverse map of `parents`.
        """
        return self._ensure_statistic(self._children)

    @property
    def scopes(self):
        """Map each graph to the complete set of graphs nested in it.

        The set associated to a graph includes the graph.
        """
        return self._ensure_statistic(self._scopes)

    @property
    def free_variables_total(self):
        """Map each graph to its free variables.

        This differs from `free_variables_direct` in that it also includes free
        variables needed by children graphs. Furthermore, graph Constants may
        figure as free variables.
        """
        return self._ensure_statistic(self._free_variables_total)

    def push_replace(self, old_node, new_node):
        """Declare replacement of old_node by new_node.

        This does not modify the graph. The change will occur in the next call
        to commit().
        """
        g = old_node.graph
        if g and g.return_ is old_node:
            raise Exception('Cannot replace the return node of a graph.')
        uses = self.uses[old_node]
        for node, key in uses:
            self.push_set_edge(node, key, new_node)

    def push_set_edge(self, node, key, value):
        """Declare setting node.inputs[key] to value.

        This does not modify the graph. The change will occur in the next call
        to commit().
        """
        self._changes.append((node, key, value))

    def commit(self):
        """Commit changes.

        This modifies the graph and update attributes and properties.
        """
        changes, self._changes = self._changes, []

        addedges = Counter()
        rmedges = Counter()

        adds = Counter()
        rms = Counter()

        for root_node, key, new_node in changes:
            old_node = root_node.inputs[key]
            rmedges[(root_node, key, old_node)] += 1
            addedges[(root_node, key, new_node)] += 1
            rms[old_node] += 1
            adds[new_node] += 1
            root_node.inputs[key] = new_node

        for root_node, key, new_node in addedges - rmedges:
            self._process_edge(root_node, key, new_node, 1)

        self._acquire_nodes(adds - rms)

        for root_node, key, old_node in rmedges - addedges:
            self._process_edge(root_node, key, old_node, -1)

        self._maybe_drop_nodes(rms - adds)
