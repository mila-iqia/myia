"""
Miscellaneous utilities go here.
"""


from typing import Iterable, Callable, List, Tuple as TupleT, TypeVar


class Props:
    """
    Create an object with attributes equal to a dictionary's keys.
    """
    def __init__(self, d):
        self.__dict__ = d

    def __getattr__(self, attr):
        return self.__dict__[attr]

    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            super().__setattr__(attr, value)
        else:
            self.__dict__[attr] = value


T = TypeVar('T')
U = TypeVar('U')


def group_contiguous(arr: Iterable[T],
                     classify: Callable[[T], U]) \
        -> List[TupleT[U, List[T]]]:
    """
    Splits the given array into groups according to the classification
    function given. Contiguous elements that have the same classification
    will be grouped together. The classification is included with each
    group.

    Example:

    >>> group_contiguous([1, 2, 4, 6, 7, 9, 10, 12], lambda x: x % 2)
    [(1, [1]), (0, [2, 4, 6]), (1, [7, 9]), (0, [10, 12])]
    """
    current_c: U = None
    results = []
    current = []
    for a in arr:
        c = classify(a)
        if current_c == c:
            current.append(a)
        else:
            if current:
                results.append((current_c, current))
            current_c = c
            current = [a]
    if current:
        results.append((current_c, current))
    return results


# TODO: document
class Keyword:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
