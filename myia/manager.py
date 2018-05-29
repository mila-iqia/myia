"""Managing graph modification and information about graphs."""


from collections import defaultdict, Counter

from .ir import succ_deeper, is_constant_graph
from .graph_utils import dfs, FOLLOW, EXCLUDE


class ParentProxy:
    """Represents a graph's immediate parent."""

    def __init__(self, graph):
        """Initialize the ParentProxy."""
        self.graph = graph

    def __hash__(self):
        return hash(self.graph)

    def __eq__(self, other):
        return isinstance(other, ParentProxy) and other.graph is self.graph


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
        roots = set(self.roots) if self.roots else set()
        self.roots = set()
        self.graphs = set()
        self.all_nodes = set()
        self.nodes = defaultdict(set)
        self.uses = defaultdict(set)
        self.free_variables_direct = defaultdict(Counter)
        self.graph_constants = defaultdict(Counter)
        self._usegraph = UseGraph()
        self.graph_dependencies_direct = defaultdict(Counter)
        self.graph_dependencies_prox = defaultdict(Counter)
        self.invalidate_nesting()
        for root in roots:
            self.add_graph(root, root=True)

    def invalidate_nesting(self):
        """Invalidate current nesting information.

        The following properties are invalidated and will be recomputed the
        next time they are requested:

        * free_variables_total
        * graph_dependencies_total
        * parents
        * children
        * scopes
        """
        self._graph_dependencies_total = None
        self._parents = None
        self._children = None
        self._scopes = None
        self._free_variables_total = None

    def clean(self):
        """Clean up properties that rely on counters.

        Some properties exported by GraphManager map graphs to a map of nodes
        to counters. This removes all mappings that map to a count of zero so
        that every node in the set of keys is guaranteed to have a positive
        count.

        This is already called automatically after commit, so there should
        normally be no need to call this method.
        """
        to_clean = [
            self.free_variables_direct,
            self._usegraph.uses,
            self._usegraph.users,
            self.graph_constants,
            self.graph_dependencies_direct,
            self.graph_dependencies_prox,
            self._free_variables_total
        ]
        for d in to_clean:
            for v in (d or {}).values():
                v._keep_positive()

    def add_graph(self, graph, root=False):
        """Add a graph to this manager, optionally as a root graph."""
        self._ensure_graph(graph)
        if root:
            self.roots.add(graph)
        self._acquire_nodes({graph.return_})

    def _ensure_graph(self, graph):
        """Ensure that the graph is managed by this manager."""
        if graph._manager and graph._manager is not self:
            raise Exception('A graph can only have one manager.')
        graph._manager = self
        self.graphs.add(graph)

    def _update_counter(self, counters, key, direction):
        """Update a counter in the given direction (1 or -1).

        If the counter changes from or to zero, nesting data is invalidated.
        """
        count = counters[key]
        counters[key] += direction
        if (count == 0) != (counters[key] == 0):
            self.invalidate_nesting()

    def _process_edge(self, node, key, inp, direction):
        """Add/remove an edge between two nodes.

        Args:
            direction:
                * 1 if the edge is added.
                * -1 if the edge is removed.
        """
        g = node.graph

        if direction == -1:
            self.uses[inp].remove((node, key))
        else:
            self.uses[inp].add((node, key))

        def _update_fvtotal(stop_graph, fv):
            tot = self._free_variables_total
            if tot is not None:
                curr = g
                while curr and curr is not stop_graph:
                    tot[curr][fv] += direction
                    curr = self.parents.get(curr, None)

        if is_constant_graph(inp):
            ig = inp.value
            self._ensure_graph(ig)
            self.graph_constants[ig][inp] += direction
            self._usegraph.update(g, ig, direction)
            self._update_counter(
                self.graph_dependencies_prox[g],
                ParentProxy(ig),
                direction
            )
            if self._parents is not None:
                p = self._parents.get(ig, None)
                if p:
                    _update_fvtotal(p, ig)

        elif inp.graph and inp.graph is not g:
            self._ensure_graph(inp.graph)
            self.free_variables_direct[g][inp] += direction
            self.graph_dependencies_direct[g][inp.graph] += direction
            self._update_counter(
                self.graph_dependencies_prox[g],
                inp.graph,
                direction
            )
            _update_fvtotal(inp.graph, inp)

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
            self.nodes[g].add(node)
            self._process_inputs(node, 1)

    def _maybe_drop_nodes(self, nodes):
        """Check if the nodes are connected to a graph, drop them if not."""
        nodes = set(nodes)

        while nodes:
            node = nodes.pop()
            g = node.graph
            assert node in self.all_nodes
            uses = self.uses[node]
            if uses or (node.graph and node is node.graph.return_):
                # This node is still live
                continue  # pragma: no cover

            self.all_nodes.remove(node)
            self.nodes[g].remove(node)
            self._process_inputs(node, -1)
            nodes.update(node.inputs)

    @property
    def graphs_used(self):
        """Map each graph to the set of graphs it uses.

        For each graph, this is the set of graphs that it refers to directly.
        """
        return self._usegraph.uses

    @property
    def graph_users(self):
        """Map each graph to the set of graphs that use it.

        For each graph, this is the set of graphs that refer to it directly.
        """
        return self._usegraph.users

    @property
    def graph_dependencies_total(self):
        """Map each graph to the set of graphs it depends on.

        This is a superset of `graph_dependencies_direct` which also
        includes the graphs from which nested graphs need free
        variables.
        """
        if self._graph_dependencies_total is not None:
            return self._graph_dependencies_total

        all_deps = self.graph_dependencies_prox

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

        new_deps = defaultdict(set)
        for g in list(all_deps.keys()):
            new_deps[g] = seek_parents(g)

        self._graph_dependencies_total = new_deps
        return new_deps

    @property
    def parents(self):
        """Map each graph to its parent graph.

        Top-level graphs are associated to `None` in the returned
        dictionary.
        """
        if self._parents is not None:
            return self._parents

        all_deps = self.graph_dependencies_total
        todo = [(g, set(deps)) for g, deps in all_deps.items()]
        todo.sort(key=lambda xy: len(xy[1]))

        parents = {}

        while todo:
            next_todo = []
            for g, deps in todo:
                if len(deps) > 1:
                    rm = set()
                    for dep in deps:
                        parent = parents.get(dep, None)
                        while parent:
                            rm.add(parent)
                            parent = parents.get(parent, None)
                    deps -= rm
                if len(deps) == 0:
                    parents[g] = None
                elif len(deps) == 1:
                    parents[g] = deps.pop()
                else:
                    next_todo.append((g, deps))
            todo = next_todo

        self._parents = parents
        return parents

    @property
    def children(self):
        """Map each graph to the graphs immediately nested in it.

        This is the inverse map of `parents`.
        """
        if self._children is not None:
            return self._children

        children = defaultdict(set)
        for g, parent in self.parents.items():
            if parent:
                children[parent].add(g)

        self._children = children
        return children

    @property
    def scopes(self):
        """Map each graph to the complete set of graphs nested in it.

        The set associated to a graph includes the graph.
        """
        if self._scopes is not None:
            return self._scopes

        scopes = defaultdict(set)
        for g in self.graphs:
            p = g
            while p:
                scopes[p].add(g)
                p = self.parents.get(p, None)

        self._scopes = scopes
        return scopes

    @property
    def free_variables_total(self):
        """Map each graph to its free variables.

        This differs from `free_variables_direct` in that it also includes free
        variables needed by children graphs. Furthermore, graph Constants may
        figure as free variables.
        """
        if self._free_variables_total is not None:
            return self._free_variables_total

        total = defaultdict(Counter)
        for g in self.graphs:
            for node, count in self.free_variables_direct[g].items():
                curr = g
                while curr:
                    total[curr][node] += count
                    curr = self.parents[curr]
                    if node in self.nodes[curr]:
                        break

            for g2, count in self.graphs_used[g].items():
                p = self.parents.get(g2, None)
                if p is None:
                    continue
                curr = g
                while curr is not p:
                    total[curr][g2] += count
                    curr = self.parents[curr]

        self._free_variables_total = total
        return total

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
        self.clean()


class UseGraph:
    """Hold graph uses and users."""

    def __init__(self):
        """Initialize UseGraph."""
        self.uses = defaultdict(Counter)
        self.users = defaultdict(Counter)

    def update(self, user, usee, direction):
        """Update relationship between user and uses.

        Args:
            direction:
                * 1: Add an occurrence of user using usee
                * -1: Remove an occurrence of user using usee
        """
        self.uses[user][usee] += direction
        self.users[usee][user] += direction
