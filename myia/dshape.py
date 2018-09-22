"""Data structures to hold shapes, used by shape inference."""


from .infer import ANYTHING, InferenceError
from .utils import Named, overload


NOSHAPE = Named('NOSHAPE')


class TupleShape:
    """Class to distinguish the shape of tuples items."""

    __slots__ = ['shape']

    def __init__(self, shape):
        """Create the shape."""
        self.shape = tuple(shape)

    def __repr__(self):
        return f"T{self.shape}"

    def __len__(self):
        return len(self.shape)

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.shape == other.shape)

    def __hash__(self):
        return hash((type(self), self.shape))


class ListShape:
    """Class to represent the shape of list elements."""

    __slots__ = ['shape']

    def __init__(self, shape):
        """Create the shape."""
        self.shape = shape

    def __repr__(self):
        return f"L{self.shape}"

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.shape == other.shape)

    def __hash__(self):
        return hash((type(self), self.shape))


class ClassShape:
    """Class to represent the shape of dataclass fields."""

    __slots__ = ['shape']

    def __init__(self, shape):
        """Create the shape."""
        self.shape = shape

    def __repr__(self):
        return f"C{self.shape}"

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.shape == other.shape)

    def __hash__(self):
        return hash((type(self), tuple(sorted(self.shape.items()))))


@overload(bootstrap=True)
def shape_cloner(self, s: TupleShape, *args):
    """Base function to clone a shape recursively.

    Create a variant of this function to make shape transformers.
    """
    return TupleShape(self(s2, *args) for s2 in s.shape)


@overload  # noqa: F811
def shape_cloner(self, s: ListShape, *args):
    return ListShape(self(s.shape, *args))


@overload  # noqa: F811
def shape_cloner(self, s: ClassShape, *args):
    return ClassShape({name: self(value, *args)
                       for name, value in s.shape.items()})


@overload  # noqa: F811
def shape_cloner(self, s: tuple, *args):
    return tuple(self(s2, *args) for s2 in s)


@overload  # noqa: F811
def shape_cloner(self, s: object, *args):
    return s


@overload
def _generalize_shape(s1: ListShape, s2):
    if not isinstance(s2, ListShape):
        raise InferenceError('Mismatched shape types')
    return ListShape(_generalize_shape(s1.shape, s2.shape))


@overload  # noqa: F811
def _generalize_shape(s1: TupleShape, s2):
    if not isinstance(s2, TupleShape):
        raise InferenceError('Mismatched shape types')
    s1 = s1.shape
    s2 = s2.shape
    if len(s1) != len(s2):
        raise InferenceError('Tuples of different lengths')
    tup = [_generalize_shape(a, b) for a, b in zip(s1, s2)]
    return TupleShape(tup)


@overload  # noqa: F811
def _generalize_shape(s1: tuple, s2):
    if not isinstance(s2, tuple):
        raise InferenceError('Mismatched shape types')
    if len(s1) != len(s2):
        raise InferenceError('Arrays of differing ndim')
    return tuple(a if a == b else ANYTHING
                 for a, b in zip(s1, s2))


@overload  # noqa: F811
def _generalize_shape(s1: ClassShape, s2):
    if not isinstance(s2, ClassShape):
        raise InferenceError('Mismatched shape types')
    d = {}
    if s1.shape.keys() != s2.shape.keys():
        raise InferenceError('Classes with different fields')
    for k, v in s1.shape.items():
        d[k] = _generalize_shape(v, s2.shape[k])


@overload  # noqa: F811
def _generalize_shape(s1: object, s2):
    if s1 == s2:
        return s1
    else:
        raise InferenceError('Cannot match shapes')


def find_matching_shape(shps):
    """Returns a shape that matches all shapes in `shps`."""
    s1, *rest = shps
    for s2 in rest:
        s1 = _generalize_shape(s1, s2)
    return s1
