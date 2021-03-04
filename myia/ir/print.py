from .node import SEQ
import io


def repr_node(node, nodecache):
    if node.is_constant_graph():
        return f"@{str(node.value)}"
    elif node.is_constant():
        return str(node.value)
    elif node.is_apply():
        return f"%{nodecache.repr(node)}"
    else:
        return f"%{str(node)}"


def _print_node(node, buf, nodecache, offset=0):
    o = " " * offset
    assert node.is_apply()
    print(f"{o}%{nodecache.repr(node)} = ", end="", file=buf)
    print(f"{repr_node(node.fn, nodecache)}(", end="", file=buf)
    print(", ".join(repr_node(a, nodecache) for a in node.inputs), end="", file=buf)
    print(")", file=buf, end="")
    if node.abstract is not None:
        print(f" ; type={node.abstract}", file=buf, end="")
    print("", file=buf)


def str_graph(g, allow_cycles=False, recursive=True):
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

        print(f"graph {str(g)}(", file=buf, end="")
        print(
            ", ".join(
                f"%{str(p)}{' : ' + str(p.astract) if p.abstract is not None else ''}"
                for p in g.parameters
            ),
            file=buf, end="",
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
            seen_nodes.add(node)
            applies.append(node)
            for edge in node.edges.values():
                if edge.node is not None:
                    if (edge.node.is_apply() and
                            edge.node.graph is g):
                        todo.append(edge.node)
                    elif recursive and edge.node.is_constant_graph():
                        todo_graphs.append(edge.node.value)

        for node in reversed(applies):
            if (node.graph is not None and node.graph is not g) or node is g.return_:
                continue
            _print_node(node, buf, nodecache, offset=2)

        print(f"  return {repr_node(g.output, nodecache)}", file=buf)
        print("}", file=buf)

    return buf.getvalue()


class NodeCache:
    def __init__(self, ctr=0):
        self.ctr = ctr
        self.cache = {}

    def repr(self, node):
        key = id(node)
        if key not in self.cache:
            self.cache[key] = f"_apply{self.ctr}"
            self.ctr += 1
        return self.cache[key]
