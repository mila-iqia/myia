"""Implementation of OrderedSet."""


class OrderedSet:
    """Like set(), but ordered."""

    def __init__(self, elems=[]):
        """Create an OrderedSet."""
        self._d = dict()
        for e in elems:
            self.add(e)

    def add(self, e):
        """Add an element."""
        self._d[e] = None

    def remove(self, e):
        """Remove an element that is present.

        Raise KeyError if not present.
        """
        del self._d[e]

    def discard(self, e):
        """Remove an element, ignore if not present."""
        self._d.pop(e, None)

    def copy(self):
        """Return a shallow copy."""
        res = OrderedSet.__new__(OrderedSet)
        res._d = self._d.copy()
        return res

    def __len__(self):
        return len(self._d)

    def __iter__(self):
        return iter(self._d)

    def __repr__(self):
        return f"OrderedSet({list(self._d)})"

    def clear(self):
        """Remove all entries."""
        self._d.clear()

    def __eq__(self, other):
        return type(self) == type(other) and self._d == other._d

    def pop(self):
        """Remove and return an element.

        Raise KeyError if empty.
        """
        return self._d.popitem()[0]

    def isdisjoint(self, other):
        """Return True if there are no elements in common."""
        return not any(e in other for e in self)

    def issubset(self, other):
        """Test whether every element in the set is in `other`."""
        return all(e in other for e in self)

    __le__ = issubset

    def __lt__(self, other):
        return self <= other and self != other

    def issuperset(self, other):
        """Test whether every element in `other` is in the set."""
        return OrderedSet(other).issubset(self)

    __ge__ = issuperset

    def __gt__(self, other):
        return self >= other and self != other

    def union(self, *others):
        """Return a new set with elements from the set and all others."""
        res = self.copy()
        res.update(*others)
        return res

    __or__ = union

    def intersection(self, *others):
        """Return a new set with elements common to the set and all others."""
        res = self.copy()
        res.intersection_update(*others)
        return res

    __and__ = intersection

    def difference(self, *others):
        """Return a new set with elements that are not in the others."""
        res = self.copy()
        res.difference_update(*others)
        return res

    __sub__ = difference

    def symmetric_difference(self, other):
        """New set with the with the elements that are not in common."""
        res = self.copy()
        res.symmetric_difference_update(other)
        return res

    __xor__ = symmetric_difference

    def update(self, *others):
        """Update the set, adding elements from all others."""
        for other in others:
            for e in other:
                self.add(e)
        return self

    __ior__ = update

    def intersection_update(self, *others):
        """Update the set, keeping only elements found in it and all others."""
        for other in others:
            for e in list(self):
                if e not in other:
                    self.discard(e)
        return self

    __iand__ = intersection_update

    def difference_update(self, *others):
        """Update the set, removing elements found in others."""
        for other in others:
            for e in other:
                self.discard(e)
        return self

    __isub__ = difference_update

    def symmetric_difference_update(self, other):
        """Update the set, keeping only the difference from both sets."""
        for e in other:
            if e in self:
                self.remove(e)
            else:
                self.add(e)

    def __contains__(self, x):
        return x in self._d


__consolidate__ = True
__all__ = ["OrderedSet"]
