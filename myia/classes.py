"""Library of Myia-compatible classes."""

from dataclasses import dataclass

from . import operations
from .operations import hyper_map
from .utils import core


class ADT:
    """Base class for an algebraic data type."""


@dataclass  # pragma: no cover
class Slice:
    """Myia version of a slice."""

    start: object
    stop: object
    step: object


@dataclass  # pragma: no cover
class Cons(ADT):
    """Cons cell for lists.

    Attributes:
        head: The first element of the list.
        tail: The rest of the list.

    """

    head: object
    tail: "Cons"

    @staticmethod
    def from_list(elems):
        """Convert a list to a linked list using Cons."""
        rval = Empty()
        for elem in reversed(elems):
            rval = Cons(elem, rval)
        return rval

    def _to_list(self):
        curr = self
        rval = []
        while not isinstance(curr, Empty):
            rval.append(curr.head)
            curr = curr.tail
        return rval

    @core(static_inline=True)
    def __bool__(self):
        return True

    def __len__(self):
        return 1 + len(self.tail)

    def __getitem__(self, idx):
        if idx == 0:
            return self.head
        else:
            return self.tail[idx - 1]

    def __iter__(self):
        return iter(self._to_list())

    @core(static_inline=True)
    def __myia_iter__(self):
        return self

    @core(static_inline=True)
    def __myia_hasnext__(self):
        return True

    @core(static_inline=True)
    def __myia_next__(self):
        return self.head, self.tail


@dataclass  # pragma: no cover
class Empty(ADT):
    """Empty list."""

    def __iter__(self):
        return iter(())

    @core(static_inline=True)
    def __bool__(self):
        return False

    @core(static_inline=True)
    def __len__(self):
        return 0

    @core(static_inline=True)
    def __getitem__(self, idx):
        raise Exception("Index out of bounds")

    @core(static_inline=True)
    def __myia_iter__(self):
        return self

    @core(static_inline=True)
    def __myia_next__(self):
        raise Exception("Out of bounds")

    @core(static_inline=True)
    def __myia_hasnext__(self):
        return False


class ArithmeticData:
    """Mixin to implement access to arithmetic operators.

    When used for a dataclass D, operations like D + D will add together
    all matching fields from the added instances.
    """

    __array_priority__ = 1_000_000

    @core
    def __add__(self, x):
        return hyper_map(operations.add, self, x)

    @core
    def __sub__(self, x):
        return hyper_map(operations.sub, self, x)

    @core
    def __mul__(self, x):
        return hyper_map(operations.mul, self, x)

    @core
    def __truediv__(self, x):
        return hyper_map(operations.truediv, self, x)

    @core
    def __floordiv__(self, x):
        return hyper_map(operations.floordiv, self, x)

    @core
    def __mod__(self, x):
        return hyper_map(operations.mod, self, x)

    @core
    def __pow__(self, x):
        return hyper_map(operations.pow, self, x)

    @core
    def __pos__(self):
        return hyper_map(operations.pos, self)

    @core
    def __neg__(self):
        return hyper_map(operations.neg, self)

    @core
    def __radd__(self, x):
        return hyper_map(operations.add, x, self)

    @core
    def __rsub__(self, x):
        return hyper_map(operations.sub, x, self)

    @core
    def __rmul__(self, x):
        return hyper_map(operations.mul, x, self)

    @core
    def __rtruediv__(self, x):
        return hyper_map(operations.truediv, x, self)

    @core
    def __rfloordiv__(self, x):
        return hyper_map(operations.floordiv, x, self)

    @core
    def __rmod__(self, x):
        return hyper_map(operations.mod, x, self)

    @core
    def __rpow__(self, x):
        return hyper_map(operations.pow, x, self)


__consolidate__ = True
__all__ = ["ADT", "ArithmeticData", "Cons", "Empty", "Slice"]
