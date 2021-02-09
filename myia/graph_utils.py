"""Generic graph utilities.

The utilities in this file should be abstract with respect to node types or
the notion of successor.
"""

from typing import Any, Callable, Dict, Iterable, List, Set, TypeVar

FOLLOW = "follow"
NOFOLLOW = "nofollow"
EXCLUDE = "exclude"


T = TypeVar("T")


def always_include(node: Any) -> str:
    """Include a node in the search unconditionally."""
    return FOLLOW


def dfs(
    root: T,
    succ: Callable[[T], Iterable[T]],
    include: Callable[[T], str] = always_include,
) -> Iterable[T]:
    """Perform a depth-first search.

    Arguments:
        root: The node to start from.
        succ: A function that returns a node's successors.
        include: A function that returns whether to include a node
            in the search.

            * Return 'follow' to include the node and follow its edges.
            * Return 'nofollow' to include the node but not follow its edges.
            * Return 'exclude' to not include the node, nor follow its edges.

    """
    seen: Set[T] = set()
    to_visit = [root]
    while to_visit:
        node = to_visit.pop()
        if node in seen:
            continue
        seen.add(node)
        incl = include(node)
        if incl == FOLLOW:
            yield node
            to_visit += succ(node)
        elif incl == NOFOLLOW:
            yield node
        elif incl == EXCLUDE:
            pass
        else:
            raise ValueError(
                "include(node) must return one of: "
                '"follow", "nofollow", "exclude"'
            )


def toposort(
    root: T,
    succ: Callable[[T], Iterable[T]],
    include: Callable[[T], str] = always_include,
    allow_cycles=False,
) -> Iterable[T]:
    """Yield the nodes in the tree starting at root in topological order.

    Arguments:
        root: The node to start from.
        succ: A function that returns a node's successors.
        include: A function that returns whether to include a node
            in the search.

            * Return 'follow' to include the node and follow its edges.
            * Return 'nofollow' to include the node but not follow its edges.
            * Return 'exclude' to not include the node, nor follow its edges.

    """
    done: Set[T] = set()
    todo: List[T] = [root]
    rank: Dict[T, int] = {}
    cycles = set()

    while todo:
        node = todo[-1]
        if node in done:
            todo.pop()
            continue
        if node in rank and rank[node] != len(todo):
            if allow_cycles:
                cycles.add(node)
            else:
                raise ValueError("cycle")
        else:
            rank[node] = len(todo)
        cont = False

        incl = include(node)
        if incl == FOLLOW:
            for i in succ(node):
                if i not in done and i not in cycles:
                    todo.append(i)
                    cont = True
        elif incl == NOFOLLOW:
            pass
        elif incl == EXCLUDE:
            done.add(node)
            todo.pop()
            continue
        else:
            raise ValueError(
                "include(node) must return one of: "
                '"follow", "nofollow", "exclude"'
            )

        if cont:
            continue
        done.add(node)
        yield node
        todo.pop()


__consolidate__ = True
__all__ = ["always_include", "dfs", "toposort"]
