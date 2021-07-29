"""Utilities to print a textual representation of graphs."""

import io
import linecache
import os
import textwrap
import types

from colorama import Fore, Style
from hrepr import pstr
from ovld.core import _Ovld

from ..utils.info import Labeler
from .node import SEQ, Constant, Node


def _print_node(node, buf, nodecache, offset=0):
    o = " " * offset
    assert node.is_apply()
    print(f"{o}{nodecache(node)} = ", end="", file=buf)
    print(f"{nodecache(node.fn)}(", end="", file=buf)
    args, kwargs = node.args()
    args = [nodecache(a) for a in args]
    args.extend(f"{k}={nodecache(v)}" for k, v in kwargs.items())

    print(
        ", ".join(args),
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

    def informative(self, obj, hide_anonymous=True, show_args=True):
        """Return a more informative label.

        The informative label for an Apply node includes the name of the
        node, the name of the function being applied and optionally its
        arguments.

        Arguments:
            obj: The Node to describe.
            hide_anonymous: Whether to hide the node's name if it was not
                given a name by the user in the source code.
            show_args: Whether to show the list of arguments.
        """
        lbl = self(obj, generate=not hide_anonymous)
        if isinstance(obj, Node) and obj.is_apply():
            fn = self(obj.fn)
            if lbl is None:
                lbl = f"→ {fn}"
            else:
                lbl = f"{lbl} = {fn}"
            if show_args:
                args = ", ".join(self(inp) for inp in obj.inputs)
                lbl = f"{lbl}({args})"
            return lbl
        else:
            return lbl

    def __call__(self, obj, generate=True):
        """Label the given object."""
        if isinstance(obj, Node) and obj.is_constant_graph():
            return super().__call__(obj.value, generate=generate)
        else:
            return super().__call__(obj, generate=generate)


global_labeler = NodeLabeler()


def _format_line(line, col1, col2, mode="color"):
    assert mode in ("caret", "color")
    leading_spaces = len(line) - len(line.lstrip())
    col1 = max(col1, leading_spaces)
    if mode == "caret":
        hl = " " * col1 + "^" * (col2 - col1) + "\n"
        return line + hl
    elif mode == "color":
        return (
            line[:col1]
            + Fore.YELLOW
            + Style.BRIGHT
            + line[col1:col2]
            + Style.RESET_ALL
            + line[col2:]
        )


class _Signature:
    def __init__(self, fn, members):
        self.fn = fn
        self.members = members

    def __hrepr__(self, H, hrepr):
        return H.bracketed(
            [
                H.pair(global_labeler(m), hrepr(m.abstract), delimiter="::")
                for m in self.members
            ],
            start=f"{global_labeler(self.fn)}(",
            end=")",
        )


def _simplified_filename(filename):
    here = os.path.abspath(".")
    if filename.startswith(here):
        filename = filename[len(here) + 1 :]
        filename = os.path.join(".", filename)
    return filename


def format_trace(trace, mode="color"):
    """Format a list of Nodes with their locations."""
    results = []
    for node in reversed(trace):
        g = getattr(node, "graph", None)
        if g:
            graph_descr = pstr(_Signature(g, g.parameters))
        else:  # pragma: no cover
            # This shouldn't happen
            graph_descr = "???"

        loc = node.debug and node.debug.find("location")
        if loc is None:
            position = "???"
            locstring = f"    {node}\n"
        else:
            position = _simplified_filename(loc.filename)
            l1, l2 = loc.line, loc.line_end

            if l1 == l2:
                position += f", line {l1}"
            else:
                position += f", lines {l1}-{l2}"
            lines = linecache.getlines(loc.filename)[l1 - 1 : l2]
            if len(lines) == 1:
                locstring = _format_line(
                    lines[0], loc.column, loc.column_end, mode=mode
                )
            else:
                ann_lines = (
                    [(lines[0], loc.column, len(lines[0]) - 1)]
                    + [(line, 0, len(line) - 1) for line in lines[1:-1]]
                    + [(lines[-1], 0, loc.column_end)]
                )
                final_lines = [
                    _format_line(*ann_line, mode=mode) for ann_line in ann_lines
                ]
                locstring = "".join(final_lines)

            locstring = textwrap.dedent(locstring)
            locstring = textwrap.indent(locstring, " " * 4)
            sz = len(str(max(l1, l2)))
            nums = list(range(l1, l2 + 1))
            if mode == "caret":
                nums = [
                    val for pair in zip(nums, [""] * len(nums)) for val in pair
                ]
            numbered_lines = [
                f"{num:{sz}} {line}"
                for num, line in zip(nums, locstring.splitlines(True))
            ]
            locstring = "".join(numbered_lines)

        results.append(f"File {position}\nIn {graph_descr}\n{locstring}")
    return "\n".join(results)


def _default_filter(trace):
    filtered = []
    last = None
    for node in trace:
        g = getattr(node, "graph", False)
        if g is not last:
            last = g
            filtered.append(node)
    return filtered


def format_exc(exc, mode="color", filter=_default_filter):
    """Format an exception that contains a Myia trace.

    If the exception has a myia_trace property, format that trace and return
    it as a string, otherwise return None.

    Arguments:
        exc: An Exception with a myia_trace field.
        mode: Either "caret" or "color".
            * "caret" lines up carets under the source code for a node.
            * "color" colors the source code for the node in bold yellow
        filter: A function that takes a list of nodes (first node is the
            closest to the exception) and returns a list of nodes for the
            traceback, potentially excluding redundant ones.
    """
    trace = getattr(exc, "myia_trace", None)
    if trace is None:
        return None

    seq = []
    while trace is not None:
        seq.append(trace.node)
        trace = trace.origin

    seq = filter(seq)

    tr = format_trace(seq, mode=mode)
    return f"{tr}\n{type(exc).__name__}: {exc}"
