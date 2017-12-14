"""Primitive operations.

Primitive operations are handled as constants in the intermediate
representation, with the constant's value being an instance of a `Primitive`
subclass.

"""


class Primitive:
    """Base class for primitives."""

    pass


class Add(Primitive):
    """Scalar addition."""

    pass
