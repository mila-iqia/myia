"""Implementation of OrderedSet."""


class OrderedSet(dict):
    """Like set(), but ordered."""
    def __init__(self, elems=[]):
        for e in elems:
            self.add(e)

    def add(self, e):
        """Add an element."""
        self[e] = None

    def remove(self, e):
        """Remove an element that is present.

        Raise KeyError if not present.
        """
        del self[e]

    def discard(self, e):
        """Remove an element, ignore if not present."""
        self.pop(e, None)

    def pop(self):
        """Remove and return an element.

        Raise KeyError if empty.
        """
        return self.popitem()[0]

    def isdisjoint(self, other):
        """Return True if there are no elements in common."""
        return not any(e in other for e in self)

    def issubset(self, other):
        """Test whether every element in the set is in `other`."""
        return all(e in other for e in self)

    def issuperset(self, other):
        """Test whether every element in `other` is in the set."""
        return other.issubset(self)

    def union(self, *others):
        """Return a new set with elements from the set and all others."""
        res = self.copy()
        res.update(others)
        return res

    def intersection(self, *others):
        """Return a new set with elements common to the set and all others."""
        res = self.copy()
        res.intersection_update(others)
        return res

    def difference(self, *others):
        """
        Return a new set with elements in the set that are not in the others.
        """
        res = self.copy()
        res.difference_update(others)
        return res

    def symmetric_difference(self, other):
        """
        Return a new set with elements in either the set or other but not both.
        """
        res = self.copy()
        res.symmetric_difference_update(other)
        return res

    def update(self, *others):
        """Update the set, adding elements from all others."""
        for other in others:
            for e in other:
                self.add(e)

    def intersection_update(self, *others):
        """Update the set, keeping only elements found in it and all others."""
        for other in others:
            for e in list(self):
                if e not in other:
                    self.discard(e)

    def difference_update(self, *others):
        """Update the set, removing elements found in others."""
        for other in others:
            for e in other:
                self.discard(e)

    def symmetric_difference_update(self, other):
        """
        Update the set, keeping only elements found in either set,
        but not in both.
        """
        for e in other:
            if e in self:
                self.remove(e)
            else:
                self.add(e)
