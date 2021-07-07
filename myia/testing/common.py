"""Common testing utilities."""
from ovld import ovld

from myia.abstract import data
from myia.abstract.to_abstract import precise_abstract


@precise_abstract.variant
def _to_abstract(self, x: type):
    return data.AbstractAtom({"interface": x})


@ovld
def _to_abstract(self, x: (data.GenericBase, data.AbstractValue)):  # noqa: F811
    return x


def A(*args):
    """Convert given arguments to an abstract value for testing."""
    if len(args) == 1:
        arg = args[0]
    else:
        arg = args
    return _to_abstract(arg)


def Un(*opts):
    """Convert given arguments to an abstract union for testing."""
    return data.AbstractUnion([A(opt) for opt in opts], tracks={})
