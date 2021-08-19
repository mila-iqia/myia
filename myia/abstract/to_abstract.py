"""Functions to convert data to an abstract data type."""
from types import ModuleType
from typing import Union

from ovld import ovld

from . import data


@ovld
def type_to_abstract(self, typ: type):
    """Convert a type to an AbstractValue."""
    return data.AbstractAtom({"interface": typ})


@ovld
def to_abstract(self, xs: tuple):
    """Convert data to an AbstractValue."""
    return data.AbstractStructure(
        [self(x) for x in xs], {"interface": type(xs)}
    )


@ovld
def to_abstract(self, x: ModuleType):  # noqa: F811
    """Convert module to an AbstractValue.

    Keep module as interface so that getattr(interface, name) is valid.
    """
    return data.AbstractAtom({"interface": x})


@ovld
def to_abstract(self, x: object):  # noqa: F811
    try:
        members = vars(x)
    except TypeError:
        members = []
    if not members:
        return data.AbstractAtom({"interface": type(x)})
    else:
        field_types = tuple(self(x) for x in members.values())
        return data.AbstractStructure(
            field_types,
            {"interface": data.TypedObject(type(x), members.keys())},
        )


@ovld
def to_abstract(self, t: type):  # noqa: F811
    return data.AbstractStructure([type_to_abstract(t)], {"interface": type})


@to_abstract.variant
def precise_abstract(self, x: Union[int, bool]):
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
