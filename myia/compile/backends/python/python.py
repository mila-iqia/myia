"""Python backend.

Myia graph does not initially provide a list of all graphs and closures it uses.
So, we need to recursively visit graph to found them. Plus, in generated code,
each closure must be correctly positioned into its parent graph,
after nodes it needs and before nodes that use it.

To handle these issues, compilation strategy consists of convert each myia graph
into a directed graph which links user to used nodes. At code generation, we can then
visit each directed graph in a reverse order, so that used nodes easily come before user nodes.
"""
import importlib
import sys
from abc import abstractmethod
from types import ModuleType

from myia.compile.backends.python.code_generator import CodeGenerator
from myia.compile.backends.python.optimizer import Optimizer
from myia.compile.backends.python.pdb_run_call import PdbRunCall
from myia.utils.directed_graph import DirectedGraph


class _GraphConverter:
    """Base class for converters."""

    @abstractmethod
    def make_arrow(self, user, used):
        """Link an node to a graph."""
        raise NotImplementedError()

    @abstractmethod
    def has_directed(self, node):
        """Return True if given node was already visited and inserted to a directed graph."""
        raise NotImplementedError()


class GraphToDirected(_GraphConverter):
    """Helper class to convert a myia graph to a directed graph."""

    def __init__(self, graph, parent: _GraphConverter):
        """Initialize.

        :param graph: myia graph
        :param parent: parent converted
        """
        self.parent = parent
        self.graph = graph
        self.directed = DirectedGraph(self.graph)
        self.todo_closures = []

    def make_arrow(self, user, used_graph):
        """Link a user node to a graph.

        Create link in current directed graph if used graph is a closure of current function.
        Otherwise, ask parent to link current function to used graph.

        :param user: user node
        :param used_graph: used graph
        """
        if user is used_graph:
            # Do not link a graph to itself.
            return
        if used_graph.parent is self.graph:
            assert self.directed.has(user)
            self.directed.add_arrow(user, used_graph)
            # Registered closure must be recursively converted later to a directed graph.
            self.todo_closures.append(used_graph)
        else:
            self.parent.make_arrow(self.graph, used_graph)

    def has_directed(self, node):
        """Return True if given myia node is in a directed graph."""
        return self.directed.has(node) or self.parent.has_directed(node)

    def _replace_directed_node(self, from_node, to_node):
        """Replace a value with another one in directed graph.

        Currently used to replace a closure with corresponding DirectedGraph instance.
        """
        if from_node in self.directed.uses:
            self.directed.uses.setdefault(to_node, [])
            to_uses = self.directed.uses[to_node]
            for used in self.directed.uses.pop(from_node):
                if used not in to_uses:
                    to_uses.append(used)
                index = self.directed.used_by[used].index(from_node)
                self.directed.used_by[used][index] = to_node
        if from_node in self.directed.used_by:
            self.directed.used_by.setdefault(to_node, [])
            to_used = self.directed.used_by[to_node]
            for user in self.directed.used_by.pop(from_node):
                if user not in to_used:
                    to_used.append(user)
                index = self.directed.uses[user].index(from_node)
                self.directed.uses[user][index] = to_node

    def generate_directed_graph(self):
        """Converted myia graph to directed graph."""
        # Generate directed graph.
        # Use (None -> graph.return_) as first arrow.
        todo_arrows = [(None, self.graph.return_)]
        while todo_arrows:
            user, node = todo_arrows.pop()
            assert user is None or user.is_apply()
            assert node.is_apply()
            # If node is registered in a parent graph, don't treat it here.
            if self.parent.has_directed(node):
                continue
            # If arrow (user -> node) is already registered, skip it, else add it.
            if not self.directed.add_arrow(user, node):
                continue
            for e in node.edges.values():
                if e.node is not None:
                    n = e.node
                    if n.is_apply():
                        todo_arrows.append((node, n))
                    elif n.is_constant_graph():
                        self.make_arrow(node, n.value)
        # Convert closures to directed graphs.
        closure_to_directed = {}
        while self.todo_closures:
            g = self.todo_closures.pop()
            if g not in closure_to_directed:
                closure_to_directed[g] = GraphToDirected(
                    g, self
                ).generate_directed_graph()
        # Replace closures with related directed graphs.
        for g, d in closure_to_directed.items():
            self._replace_directed_node(g, d)
        # Return directed graph. ALl nodes are either apply nodes or directed graphs.
        return self.directed


class GraphToModule(_GraphConverter):
    """Helper class to convert a graph to a module.

    Used to collect and compile all module graphs related to a given graph.
    """

    def __init__(self):
        self.todo_graphs = []

    def has_directed(self, node):
        """No node can be registered at this level."""
        return False

    # NB: This method is never called currently.
    # This means we only compile one module graph (graph with no parent) per compilation,
    # and we never encounter another module graph called by main compiled graph.
    # I guess this could change in the future ?
    def make_arrow(self, user, graph):  # pragma: no cover
        """Collect graph at module level."""
        assert user is self, user
        assert graph.parent is None
        self.todo_graphs.append(graph)

    def generate_directed_graphs(self, g):
        """Collect and convert all graphs to directed graphs.

        :param g: myia graph to compile.
        :return: list of directed graphs, including given graph and all related module graphs.
        """
        self.todo_graphs.append(g)
        seen_graphs = set()
        directed_graphs = []
        while self.todo_graphs:
            graph = self.todo_graphs.pop()
            if graph not in seen_graphs:
                seen_graphs.add(graph)
                directed_graphs.append(
                    GraphToDirected(graph, self).generate_directed_graph()
                )
        return directed_graphs


class PythonBackend:
    """Python backend main class."""

    def __init__(self, debug=False, pdb=False, optimize=True):
        """Initialize.

        :param debug: if False or None, do nothing.
            If True, print generated code in stdout.
            Otherwise, should be an output stream (e.g. stdout or a StringIO)
            and generated code will be written into given stream.
        :param pdb: if True, compiled function will be run in a pdb instance
        :param optimize: if True, apply Python compilation optimizer before generating the code.
        """
        if debug:
            debug = sys.stdout if debug is True else debug
            assert hasattr(debug, "write")

        self.debug = debug
        self.pdb = bool(pdb)
        self.optimize = bool(optimize)

    @classmethod
    def nested_list_to_code_string(cls, structure, indentation=""):
        """Convert a nested list of strings to a correctly indented Python code."""
        code = ""
        for index_entry, entry in enumerate(structure):
            if not isinstance(entry, list):
                code += f"{indentation}{entry}\n" if entry else "\n"
            elif entry:
                # Indent.
                code += cls.nested_list_to_code_string(
                    entry, indentation + "  "
                )
                # Add an extra-line if there is still code to write.
                if (
                    index_entry + 1 < len(structure)
                    and structure[index_entry + 1]
                ):
                    code += "\n"
        return code

    @classmethod
    def generate_static_import(cls, value, import_name):
        """Generate a static import string if possible.

        :param value: value to import in static import
        :param import_name: name to import value (`value as name`)
        :return: static import if possible, else None
        """
        if isinstance(value, ModuleType):
            package = value.__package__
            name = value.__name__
            if importlib.import_module(name, package) is value:
                return (
                    (f"from {package} " if package else "")
                    + f"import {name}"
                    + (f" as {import_name}" if name != import_name else "")
                )
        elif hasattr(value, "__module__"):
            modname = value.__module__
            name = value.__name__
            qualname = value.__qualname__
            if (
                "." not in qualname
                and getattr(importlib.import_module(modname), name) is value
            ):
                return f"from {modname} import {name}" + (
                    f" as {import_name}" if name != import_name else ""
                )
        return None

    def compile(self, graph):
        """Compile given graph.

        :param graph: myia graph to compile
        :return: executable
        """
        code = []
        directed_graphs = GraphToModule().generate_directed_graphs(graph)
        code_generator = CodeGenerator(
            **(Optimizer().optimize(directed_graphs) if self.optimize else {})
        )
        for directed in directed_graphs:
            code.extend(code_generator.directed_graph_to_code(directed))

        static_imports = []
        runtime_imports = []
        module = {}
        # Generate static imports if possible.
        # Otherwise, symbols will be dynamically imported.
        for name, value in code_generator.globals.items():
            static_import = self.generate_static_import(value, name)
            if static_import:
                static_imports.append(static_import)
            else:
                runtime_imports.append(f"# Dynamic external import: {name}")
                # Collect dynamic imports.
                module[name] = value

        dynamic_imports = static_imports + runtime_imports
        final_structure = (
            dynamic_imports + ([""] if dynamic_imports else []) + code
        )
        final_code = self.nested_list_to_code_string(final_structure)

        if self.debug:
            self.debug.write(final_code)

        if self.pdb:
            return PdbRunCall(final_code, code_generator.label(graph), module)

        # Compile code string to a Python executable function
        # reference: https://stackoverflow.com/a/19850183
        compiled = compile(final_code, "", "exec")
        exec(compiled, module)

        return module[code_generator.label(graph)]


def compile_graph(graph, **options):
    """Helper function to quickly compile a myia graph.

    :param graph: myia graph
    :param options: Python backend options
    :return: callable (compiled function)
    """
    return PythonBackend(**options).compile(graph)
