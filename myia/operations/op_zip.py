"""Implementation of the 'zip' operation."""

from dataclasses import dataclass

from ..lib import core
from ..operations import myia_hasnext, myia_iter, myia_next


@dataclass
class Zip2:  # pragma: no cover
    """Implement zip with two arguments."""

    iter1: object
    iter2: object

    @core
    def __len__(self):
        return len(self.iter1)

    @core
    def __myia_iter__(self):
        return self

    @core
    def __myia_next__(self):
        nxt1, iter1 = myia_next(self.iter1)
        nxt2, iter2 = myia_next(self.iter2)
        return (nxt1, nxt2), Zip2(iter1, iter2)

    @core
    def __myia_hasnext__(self):
        return myia_hasnext(self.iter1) and myia_hasnext(self.iter2)


@core
def zip_(seq1, seq2):
    """Myia implementation of the standard zip function."""
    return Zip2(myia_iter(seq1), myia_iter(seq2))


__operation_defaults__ = {
    "name": "zip",
    "registered_name": "zip",
    "mapping": zip_,
    "python_implementation": zip,
}
