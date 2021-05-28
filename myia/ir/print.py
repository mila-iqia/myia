"""Utilities to print a textual representation of graphs."""

import io
import types

from ovld.core import _Ovld

from ..utils.info import Labeler
from .node import SEQ, Constant, Node


def _print_node(node, buf, nodecache, offset=0):
    o = " " * offset
    assert node.is_apply()
    print(f"{o}{nodecache(node)} = ", end="", file=buf)
    print(f"{nodecache(node.fn)}(", end="", file=buf)
    print(
        ", ".join(nodecache(a) for a in node.inputs),
        end="",
        file=buf,
    )
    print(")", file=buf, end="")
    if node.abstract is not None:
        print(f" ; type={node.abstract}", file=buf, end="")
    print("", file=buf)


def str_graph(g, allow_cycles=False, recursive=True):
    """Return a textual representation of a graph.

    Arguments:
       g: The graph to print
       allow_cycles: Useful for debugging broken graphs that contain cycles
       recursive: Also print subgraphs.
    """
    nodecache = NodeLabeler()
    buf = io.StringIO()
    seen_nodes = set()
    seen_graphs = set()
    todo_graphs = [g]

    applies = []
    first = True

    while todo_graphs:
        g = todo_graphs.pop()
        if g in seen_graphs:
            continue
        seen_graphs.add(g)

        if not first:
            print("", file=buf)
        first = False

        print(f"graph {nodecache(Constant(g))}(", file=buf, end="")
        print(
            ", ".join(
                f"{nodecache(p)}{': ' + str(p.abstract) if p.abstract is not None else ''}"
                for p in g.parameters
            ),
            file=buf,
            end="",
        )

        print(") ", file=buf, end="")
        if g.return_.abstract is not None:
            print(f"-> {g.return_.abstract} ", file=buf, end="")
        print("{", file=buf)

        todo = [g.return_]
        while todo:
            node = todo.pop()
            if node in seen_nodes:
                if allow_cycles:
                    continue
                else:
                    raise ValueError("cycle in sequence edges")
            # This can happen due to replaces
            if not node.is_apply():
                continue
            seen_nodes.add(node)
            applies.append(node)

            seq = node.edges.get(SEQ, None)
            if seq:
                todo.append(seq.node)
            if recursive:
                nodes = [e.node for e in node.edges.values()]
                for node in nodes:
                    if node.is_constant_graph():
                        todo_graphs.append(node.value)

        def _print_stragglers(n):
            nodes = [e.node for e in n.edges.values()]
            for node in nodes:
                if node.is_apply() and node not in seen_nodes:
                    seen_nodes.add(node)
                    _print_stragglers(node)
                    _print_node(node, buf, nodecache, offset=2)

        for node in reversed(applies):
            if node.graph is not None and node.graph is not g:
                continue
            _print_stragglers(node)
            if node is not g.return_:
                _print_node(node, buf, nodecache, offset=2)

        print(f"  return {nodecache(g.output)}", file=buf)
        print("}", file=buf)

    return buf.getvalue()


class NodeLabeler(Labeler):
    """Adapter for the Labeler to deal with Constant graphs."""

    def disambiguate(self, label, id):
        """Disambiguate identical labels."""
        return f"{label}.{id}"

    def describe_object(self, node):
        """Describe constants."""
        if not isinstance(node, Node) or node.is_constant_graph():
            return None
        elif node.is_constant(
            (
                type,
                types.FunctionType,
                types.BuiltinFunctionType,
                types.BuiltinMethodType,
                _Ovld,
            )
        ):
            f = node.value
            m = f.__module__
            if m == "builtins":
                return f.__qualname__
            else:
                return f"{m}.{f.__qualname__}"
        elif node.is_constant():
            return repr(node.value)
        else:
            return None

    def __call__(self, obj):
        """Label the given object."""
        if isinstance(obj, Node) and obj.is_constant_graph():
            return super().__call__(obj.value)
        else:
            return super().__call__(obj)


global_labeler = NodeLabeler()
