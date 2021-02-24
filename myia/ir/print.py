from .node import SEQ
import io


def repr_node(node):
    if node.is_constant_graph():
        return f"@{str(node.value)}"
    elif node.is_constant():
        return str(node.value)
    else:
        return f"%{str(node)}"


def _print_node(node, buf, offset=0):
    o = " " * offset
    assert node.is_apply()
    print(f"{o}%{str(node)} = ", end="", file=buf)
    print(f"{repr_node(node.inputs[0])}(", end="", file=buf)
    print(", ".join(repr_node(a) for a in node.inputs), end="", file=buf)
    print(")", file=buf, end="")
    if node.abstract is not None:
        print(f" ; type={node.abstract}", file=buf, end="")
    print("", file=buf)


def str_graph(g, allow_cycles=False):
    buf = io.StringIO()
    print(f"graph {str(g)}(", file=buf, end="")
    print(
        ",".join(
            f"%{str(p)}{' : ' + str(p.astract) if p.abstract is not None else ''}"
            for p in g.parameters
            ),
        file=buf, end="",
        )

    print(") ", file=buf, end="")
    if g.return_.abstract is not None:
        print(f"-> {g.return_.abstract} ", file=buf, end="")
    print("{", file=buf)

    seen_nodes = set()

    node = g.output
    applies = []

    while node is not None and node.is_apply():
        if node in seen_nodes:
            if allow_cycles:
                break
            else:
                raise ValueError("cycle in sequence edges")
        seen_nodes.add(node)
        applies.append(node)
        if SEQ in node.edges:
            node = node.edges[SEQ].node
        else:
            node = None

    for node in reversed(applies):
        if (node.graph is not None and node.graph is not g) or node is g.return_:
            continue
        _print_node(node, buf, offset=2)

    print(f"  return %{str(g.output)}", file=buf)
    print("}", file=buf)
    return buf.getvalue()
