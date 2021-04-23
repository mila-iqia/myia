from .node import SEQ
import io
from ..utils.info import get_debug
from collections import Counter
from .node import Constant


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
    if get_debug():
        nodecache = Labeller()
    else:
        nodecache = NodeCache()
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
                f"{nodecache(p)}{' : ' + str(p.astract) if p.abstract is not None else ''}"
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
            if seq and seq.node is not None:
                todo.append(seq.node)
            if recursive:
                nodes = [e.node for e in node.edges.values()]
                for node in nodes:
                    if node is not None and node.is_constant_graph():
                        todo_graphs.append(node.value)

        def _print_stragglers(n):
            nodes = [e.node for e in n.edges.values()]
            for node in nodes:
                if (
                    node is not None
                    and node.is_apply()
                    and node not in seen_nodes
                ):
                    seen_nodes.add(node)
                    _print_stragglers(node)
                    _print_node(node, buf, nodecache, offset=2)

        for node in reversed(applies):
            if (
                node.graph is not None and node.graph is not g
            ) or node is g.return_:
                continue
            _print_stragglers(node)
            _print_node(node, buf, nodecache, offset=2)

        print(f"  return {nodecache(g.output)}", file=buf)
        print("}", file=buf)

    return buf.getvalue()


class NodeCache:
    def __init__(self):
        self.cache = {}
        self.names = Counter()

    def __call__(self, node):
        key = id(node)
        if node.is_constant():
            key = id(node.value)
        if key not in self.cache:
            if node.is_apply():
                name = f"%_apply"
            elif node.is_constant_graph():
                name = f"@{str(node.value)}"
            elif node.is_constant():
                name = str(node.value)
            else:
                name = f"%{str(node)}"

            n = self.names[name]
            self.names[name] += 1
            if n == 0:
                if name == '%_apply':
                    name = '%_apply0'
            else:
                name = name + str(n)
            self.cache[key] = name
        return self.cache[key]
