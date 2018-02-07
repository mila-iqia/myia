"""Generic graph utilities.

The utilities in this file should be abstract with respect to node types or
the notion of successor.
"""


from typing import Callable, Iterable, Set, TypeVar


T = TypeVar('T')


def dfs(root: T, succ: Callable[[T], Iterable[T]]) -> Iterable[T]:
    """Perform a depth-first search.

    Arguments:
        root: The node to start from.
        succ: A function that returns a node's successors.
    """
    seen: Set[T] = set()
    to_visit = [root]
    while to_visit:
        node = to_visit.pop()
        if node in seen:
            continue
        seen.add(node)
        yield node
        to_visit += succ(node)
