
from typing import Iterable, Callable, List, Tuple as TupleT, TypeVar


class Props:
    """
    Create an object with attributes equal to a dictionary's keys.
    """
    def __init__(self, d):
        self.__dict__ = d

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, item, value):
        self.__dict__[item] = value

    def __getattr__(self, attr):
        return self.__dict__[attr]

    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            super().__setattr__(attr, value)
        else:
            self.__dict__[attr] = value


class SymbolsMeta(type):
    def __new__(cls, name, bases, attributes):
        return Props({k: v for k, v in attributes.items()
                      if not k.startswith('_')})


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
class Singleton:
    __instance__ = None

    def __new__(cls):
        if cls.__instance__ is not None:
            return cls.__instance__
        else:
            inst = super().__new__(cls)
            cls.__instance__ = inst
            return inst

    def __str__(self):
        return self.__class__.__name__

    __repr__ = __str__
