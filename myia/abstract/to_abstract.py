"""Functions to convert data to an abstract data type."""

from ovld import ovld

from . import data


@ovld
def to_abstract(self, xs: tuple):
    """Convert data to an AbstractValue."""
    return data.AbstractStructure(
        [self(x) for x in xs], {"interface": type(xs)}
    )


@ovld
def to_abstract(self, x: object):  # noqa: F811
    return data.AbstractAtom({"interface": type(x)})


@to_abstract.variant
def precise_abstract(self, x: (int, bool)):
    """Convert data to an AbstractValue.

    Unlike to_abstract this keeps values in the type for ints and bools.
    """
    return data.AbstractAtom({"value": x, "interface": type(x)})


def from_value(value, broaden=False):
    """Convert data to an AbstractValue.

    Arguments:
        value: The Python object to convert to an AbstractValue.
        broaden: Whether to keep the values of ints and bools in the resulting
            types.
    """
    if broaden:
        return to_abstract(value)
    else:
        return precise_abstract(value)
