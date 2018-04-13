"""Graph analysis framework."""
from typing import Any, Callable, Dict, Iterable, TypeVar
from functools import wraps

from myia.anf_ir import Graph, Apply, Constant, ANFNode
from myia.anf_ir_utils import is_constant_graph
from myia.graph_utils import toposort
from myia.prim.ops import return_
from myia.unify import DomainUnification, Var


T = TypeVar('T')


def is_return(node: ANFNode) -> bool:
    """Check if a node is a return node."""
    return (isinstance(node, Apply) and
            isinstance(node.inputs[0], Constant) and
            node.inputs[0].value == return_)


class Plugin:
    """Base class for analyzer plugins.

    Equivalence is defined by default to be type equivalence, meaning
    that all objects of the same type are considered to be equal.  If
    this is not right for your plugin, feel free to override this.
    """

    NAME: str
    analyzer: 'GraphAnalyzer'

    def __eq__(self, other):
        return type(self) == type(other)

    def register(self, analyzer: 'GraphAnalyzer') -> None:
        """Attach the plugin to an analyzer."""
        assert not hasattr(self, 'analyzer')
        self.analyzer = analyzer
        self.on_attach()

    def on_attach(self) -> None:
        """Override this method to prepare for analysis.

        This can include registering hooks, adding method shortcuts or
        adding plugin dependencies.
        """

    def on_preprocess(self) -> None:
        """This is called at the beginning of the graph traversal."""

    def on_node(self, node: ANFNode) -> Any:
        """This is called once for each node in the graphs."""

    def on_graph(self, graph: Graph) -> Any:
        """This is called once for every graph."""

    def on_postprocess(self) -> None:
        """This is called once every node has been visited."""


def event(collect: bool) -> Callable[..., Callable]:
    """Make an event function that propagates to the plugins."""
    def deco(fn: Callable) -> Callable:
        if collect:
            @wraps(fn)
            def res_collect(self, *args, **kwargs):
                res = dict()
                for n, h in self._plugins.items():
                    res[n] = getattr(h, fn.__name__)(*args, **kwargs)
                return res
            return res_collect
        else:
            @wraps(fn)
            def res_simple(self, *args, **kwargs):
                for n, h in self._plugins.items():
                    getattr(h, fn.__name__)(*args, **kwargs)
        return res_simple
    return deco


class PluginManager:
    """Helper class to manage a collection of plugins."""

    def __init__(self) -> None:
        """Create a collection of plugins."""
        self._plugins: Dict[str, Plugin] = dict()

    @event(collect=False)
    def on_preprocess(self) -> None:
        """Called when initiating an analysis."""
        pass  # pragma: no cover

    @event(collect=True)
    def on_node(self, node: ANFNode) -> Any:
        """Called for each new node."""
        pass  # pragma: no cover

    @event(collect=True)
    def on_graph(self, graph: Graph) -> Any:
        """Called for each new graph."""
        pass  # pragma: no cover

    @event(collect=False)
    def on_postprocess(self) -> None:
        """Called when finishing an analysis."""
        pass  # pragma: no cover

    def add(self, plugin: Plugin) -> None:
        """Add a plugin to the collection."""
        assert plugin.NAME not in self._plugins
        self._plugins[plugin.NAME] = plugin

    def __contains__(self, value):
        return value in self._plugins

    def __getitem__(self, key):
        return self._plugins[key]

    def __len__(self):
        return len(self._plugins)


class GraphAnalyzer:
    """Process a graph from inputs to return to perform some analyses."""

    def __init__(self, plugins: Iterable[Plugin]) -> None:
        """Create the basic attributes and register the plugins."""
        self.graphs: Dict[Graph, Dict[str, Any]] = dict()
        self._info_map: Dict[ANFNode, Dict[str, Any]] = dict()
        self._shortcuts: Dict[str, Any] = dict()
        self.equiv: Dict[Var, Any] = dict()
        self.DU = DomainUnification(dict())

        self.plugins = PluginManager()

        for p in plugins:
            self.add_plugin(p)

    def add_plugin(self, plugin: Plugin) -> None:
        """Register a plugin.

        Note that this method cannot be called once analysis has begun
        since it would lead to bad internal state.
        """
        if plugin.NAME in self.plugins:
            assert self.plugins[plugin.NAME] == plugin
            return

        assert len(self._info_map) == 0

        plugin.register(self)

        self.plugins.add(plugin)

        if hasattr(plugin, 'visit'):
            self.DU.add_domain(plugin.NAME, plugin.visit)  # type: ignore

    def add_shortcut(self, name: str, value: Any):
        """Add an extra "shortcut" attribute to the graph."""
        if name.startswith('_'):
            raise ValueError("Shortcuts for private attributes not allowed")
        if name in self._shortcuts:
            raise ValueError(f"Duplicate shortcut {name}")
        if name in self.__dict__ or name in type(self).__dict__:
            raise ValueError(f"Cannot replace builtin attribute {name}")
        self._shortcuts[name] = value

    def __getattr__(self, name):
        if name in self._shortcuts:
            return self._shortcuts[name]
        raise AttributeError(name)

    def _handle_graph(self, graph: Graph) -> None:
        for p in graph.parameters:
            if p not in self._info_map:
                self._info_map[p] = self.plugins.on_node(p)
        self.graphs[graph] = self.plugins.on_graph(graph)

    def analyze(self, graph: Graph) -> None:
        """Analyze a graph.

        This may be a graph that was previously analyzed in which case
        only the new nodes will be analyzed.  If a graph has been
        entirely analyzed before, this is a no-op.
        """
        self.plugins.on_preprocess()

        if graph not in self.graphs:
            self._handle_graph(graph)

        self._analyze_node(graph.return_)
        self.plugins.on_postprocess()

    def analyze_node(self, root_node: ANFNode) -> None:
        """Analyze a node tree.

        This will run analyses on the passed node and all of its
        children, including called graphs.  Any node that was
        previously visited will skip analysis, but still be visited to
        find reachable nodes.
        """
        self.plugins.on_preprocess()
        self._analyze_node(root_node)
        self.plugins.on_postprocess()

    def _analyze_node(self, root_node: ANFNode) -> None:
        def _succ(node):
            if is_constant_graph(node) and node.value not in self.graphs:
                g = node.value
                self._handle_graph(g)
                yield g.return_

            for i in node.inputs:
                yield i

        for node in toposort(root_node, _succ):
            if node not in self._info_map:
                self._info_map[node] = self.plugins.on_node(node)
