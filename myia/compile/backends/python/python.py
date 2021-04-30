"""Python backend.

Current compilation strategy consists of:
- convert myia graph to a directed graph of applies and closures.
  - Link (apply -> apply_or_closure) if apply uses apply_or_closure.
  - All graphs and closures are recursively converted.
- visit directed graphs recursively.
- for each graph, use visited nodes in reverse order to compile function.
"""

import sys
from abc import abstractmethod

from myia.compile.backends.python.code_generator import CodeGenerator
from myia.compile.backends.python.directed_graph import DirectedGraph
from myia.compile.backends.python.pdb_run_call import PdbRunCall


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

    def generate_directed_graph(self):
        """Converted myia graph to directed graph."""
        # Generate directed graph.
        # Use (None -> graph.return_) as first arrow.
        todo_arrows = [(None, self.graph.return_)]
        while todo_arrows:
            user, node = todo_arrows.pop()
            assert node.is_apply()
            if self.has_directed(node):
                continue
            self.directed.add_arrow(user, node)
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
            self.directed.replace(g, d)
        # Return directed graph. ALl nodes are either apply nodes or directed graphs.
        return self.directed


class GraphToModule(_GraphConverter):
    """Helper class to convert a graph to a module.

    Used to collect and compile all module graphs related to a given graph.
    """

    def __init__(self):
        """Initialize."""
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

    def __init__(self, debug=False, pdb=False):
        """Initialize.

        :param debug: if False or None, do nothing.
            If True, print generated code in stdout.
            Otherwise, should be an output stream (e.g. stdout or a StringIO)
            and generated code will be written into given stream.
        :param pdb: if True, compiled function will be run in a pdb instance
        """
        if debug:
            debug = sys.stdout if debug is True else debug
            assert hasattr(debug, "write")

        self.debug = debug
        self.pdb = bool(pdb)

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

    def compile(self, graph):
        """Compile given graph.

        :param graph: myia graph to compile
        :return: executable
        """
        code_generator = CodeGenerator()
        code = []
        for directed in GraphToModule().generate_directed_graphs(graph):
            code.extend(code_generator.directed_graph_to_code(directed))

        module = code_generator.globals
        dynamic_imports = [
            f"# Dynamic external import: {name}" for name in module
        ]
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
