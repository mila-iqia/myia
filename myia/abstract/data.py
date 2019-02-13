"""Data structures to represent data in an abstract way, for inference."""


from typing import Tuple
from dataclasses import dataclass
from contextvars import ContextVar

from .. import dtype
from ..debug.label import label
from ..utils import Named, Partializable, eprint

from .loop import Pending


# Represents the absence of inferred data
ABSENT = Named('ABSENT')

# Represents an unknown value
ANYTHING = Named('ANYTHING')

# Represents inference problems
VOID = Named('VOID')

# Represents specialization problems
DEAD = Named('DEAD')
POLY = Named('POLY')


#####################
# Function wrappers #
#####################


class Possibilities(frozenset):
    """Represents a set of possible values."""


@dataclass(frozen=True)
class Function:
    """Represents a possible function in an AbstractFunction."""


@dataclass(frozen=True)
class GraphFunction(Function):
    """Represents a Graph in a certain Context.

    Attributes:
        graph: The graph
        context: The context, or Context.empty()
        tracking_id: Identifies different uses of the same graph/context pair.

    """

    graph: 'Graph'
    context: 'Context'
    tracking_id: object = None


@dataclass(frozen=True)
class PrimitiveFunction(Function):
    """Represents a Primitive.

    Attributes:
        prim: The primitive
        tracking_id: Identifies different uses of the same primitive.

    """

    prim: 'Primitive'
    tracking_id: object = None


@dataclass(frozen=True)
class MetaGraphFunction(Function):
    """Represents a MetaGraph in a certain Context.

    Attributes:
        metagraph: The metagraph
        context: The context, or Context.empty()
        tracking_id: Identifies different uses of the same metagraph.

    """

    metagraph: 'MetaGraph'
    context: 'Context'
    tracking_id: object = None


@dataclass(frozen=True)
class PartialApplication(Function):
    """Represents a partial application.

    Attributes:
        fn: A Function
        args: The first few arguments of that function

    """

    fn: Function
    args: Tuple['AbstractBase']


@dataclass(frozen=True)
class JTransformedFunction(Function):
    """Represents a Function transformed through the application of J.

    Attributes:
        fn: A Function

    """

    fn: object


@dataclass(frozen=True)
class VirtualFunction(Function):
    """Represents some function with an explicitly given type signature.

    Attributes:
        args: The abstract arguments given to the function
        output: The abstract output

    """

    args: Tuple['AbstractBase']
    output: 'AbstractBase'


@dataclass(frozen=True)
class TypedPrimitive(Function):
    """Represents a Primitive with an explicitly given type signature.

    Attributes:
        prim: The Primitive
        args: The abstract arguments given to the Primitive
        output: The abstract output

    """

    prim: 'Primitive'
    args: Tuple['AbstractBase']
    output: 'AbstractBase'


class DummyFunction(Function):
    """Represents a function that can't be called."""


#################
# Abstract data #
#################


class AbstractBase:
    """Base class for abstract data."""

    def key(self):
        """Return a key for hash/equality purposes."""
        if not hasattr(self, '_key'):
            self._key = self._make_key()
        return self._key

    def _make_key(self):
        raise NotImplementedError()

    def __eq__(self, other):
        return type(self) is type(other) \
            and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())


class AbstractValue(AbstractBase):
    """Base class for all abstract values.

    Attributes:
        values: A dictionary mapping a Track like VALUE or TYPE
            to a value for that track. Different abstract structures
            may have different tracks, e.g. SHAPE for arrays.

    """

    def __init__(self, values):
        """Initialize an AbstractValue."""
        self.values = TrackDict(values)

    def _make_key(self):
        return tuple(sorted(self.values.items()))


class AbstractScalar(AbstractValue):
    """Represents a scalar (integer, float, bool, etc.)."""

    def __repr__(self):
        contents = [f'{k}={v}' for k, v in self.values.items()
                    if v not in (ABSENT, ANYTHING)]
        return f'S({", ".join(contents)})'


class AbstractType(AbstractValue):
    """Represents a type as a first class value."""

    def __init__(self, typ):
        """Initialize an AbstractType."""
        super().__init__({VALUE: typ})

    def __repr__(self):
        return f'Ty({self.values[VALUE]})'


class AbstractError(AbstractValue):
    """Represents some kind of error in the computation."""

    def __init__(self, err):
        """Initialize an AbstractError."""
        super().__init__({VALUE: err})

    def __repr__(self):
        return f'E({self.values[VALUE]})'


class AbstractFunction(AbstractValue):
    """Represents a function or set of functions.

    The VALUE track for an AbstractFunction contains a Possibilities object
    which is a set of Functions that might be called at this point. These
    functions must all return the same type of abstract data when called with
    the same arguments.

    Instead of a set of Possibilities, the VALUE can also be Pending.
    """

    def __init__(self, *poss, value=None):
        """Initialize an AbstractFunction.

        Provide either *poss or value, not both.

        Arguments:
            poss: Possible Functions that could be called here.
            value: Either Possibilities or Pending.
        """
        assert (len(poss) > 0) ^ (value is not None)
        v = Possibilities(poss) if value is None else value
        super().__init__({VALUE: v})

    async def get(self):
        """Return a set of all possible Functions (asynchronous)."""
        v = self.values[VALUE]
        return (await v if isinstance(v, Pending) else v)

    def get_sync(self):
        """Return a set of all possible Functions (synchronous)."""
        return self.values[VALUE]

    def get_unique(self):
        """If there is exactly one possible function, return it.

        Otherwise, raise a MyiaTypeError.
        """
        poss = self.values[VALUE]
        if isinstance(poss, Pending):  # pragma: no cover
            # This is a bit circumstantial and difficult to test explicitly
            raise MyiaTypeError('get_unique invalid because Pending')
        if len(poss) != 1:
            raise MyiaTypeError(f'Expected unique function, not {poss}')
        fn, = poss
        return fn

    def __repr__(self):
        return f'Fn({self.values[VALUE]})'


class AbstractTuple(AbstractValue):
    """Represents a tuple of elements."""

    def __init__(self, elements, values=None):
        """Initialize an AbstractTuple."""
        super().__init__(values or {})
        self.elements = tuple(elements)

    def _make_key(self):
        elms = tuple(e._make_key() for e in self.elements)
        return (super()._make_key(), elms)

    def __repr__(self):
        return f'T({", ".join(map(repr, self.elements))})'


class AbstractArray(AbstractValue):
    """Represents an array.

    The SHAPE track on an array contains the array's shape.

    Arrays must be homogeneous, hence a single AbstractValue is used to
    represent every element.

    Attributes:
        element: AbstractValue representing each element of the array.

    """

    def __init__(self, element, values):
        """Initialize an AbstractArray."""
        super().__init__(values)
        self.element = element

    def _make_key(self):
        return (super()._make_key(), self.element._make_key())

    def __repr__(self):
        return f'A({self.element}, SHAPE={self.values[SHAPE]})'


class AbstractList(AbstractValue):
    """Represents a list.

    Lists must be homogeneous, hence a single AbstractValue is used to
    represent every element.

    Attributes:
        element: AbstractValue representing each element of the list.

    """

    def __init__(self, element, values=None):
        """Initialize an AbstractList."""
        super().__init__(values or {})
        self.element = element

    def _make_key(self):
        return (super()._make_key(), self.element._make_key())

    def __repr__(self):
        return f'L({self.element})'


class AbstractClass(AbstractValue):
    """Represents a class, typically those defined using @dataclass.

    Attributes:
        tag: The class's name (a Named instance).
        attributes: Maps each field name to a corresponding AbstractValue.
        methods: Maps method names to corresponding functions, which will
            be parsed and converted by the engine when necessary, with the
            instance as the first argument.

    """

    def __init__(self, tag, attributes, methods, values={}):
        """Initialize an AbstractClass."""
        super().__init__(values)
        self.tag = tag
        self.attributes = attributes
        self.methods = methods

    def _make_key(self):
        attrs = tuple((k, v._make_key()) for k, v in self.attributes.items())
        return (super()._make_key(), self.tag, attrs)

    def __repr__(self):
        elems = [f'{k}={v}' for k, v in self.attributes.items()]
        return f'*{self.tag}({", ".join(elems)})'


class AbstractJTagged(AbstractValue):
    """Represents a value (non-function) transformed through J."""

    def __init__(self, element):
        """Initialize an AbstractJTagged."""
        super().__init__({})
        self.element = element

    def _make_key(self):
        return (super()._make_key(), self.element._make_key())

    def __repr__(self):
        return f'J({self.element})'


##########
# Tracks #
##########


class TrackDict(dict):
    """Mapping from a Track to a value."""


class Track:
    """Represents a type of property of an abstract value."""

    def __init__(self, name):
        """Initialize the Track."""
        self.name = name

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name

    def clone(self, v, recurse):
        """Clone the value associated to this Track in a TrackDict."""
        return recurse(v)

    async def async_clone(self, v, recurse):
        """Clone the value associated to this Track in a TrackDict.

        This is an asynchronous version of clone.
        """
        return await recurse(v)

    def broaden(self, v, recurse, loop):
        """Make a value more generic.

        By default, this amounts to a straight copy.
        """
        return recurse(v, loop)


class _ValueTrack(Track):
    """Represents the VALUE track."""

    def broaden(self, v, recurse, loop):
        """Values are broadened to ANYTHING."""
        return ANYTHING


class _TypeTrack(Track):
    """Represents the TYPE track, for scalars."""


class _ShapeTrack(Track):
    """Represents the SHAPE track, for arrays."""


VALUE = _ValueTrack('VALUE')
TYPE = _TypeTrack('TYPE')
SHAPE = _ShapeTrack('SHAPE')


##########
# Errors #
##########


infer_trace = ContextVar('infer_trace')
infer_trace.set({})


class Unspecializable(Exception):
    """Raised when it is impossible to specialize an inferrer."""

    def __init__(self, problem):
        """Initialize Unspecializable."""
        problem = dtype.Problem[problem]
        super().__init__(problem)
        self.problem = problem


class InferenceError(Exception, Partializable):
    """Inference error in a Myia program.

    Attributes:
        message: The error message.
        refs: A list of references which are involved in the error,
            e.g. because they have the wrong type or don't match
            each other.
        traceback_refs: A map from a context to the first reference in
            that context that fails to resolve because of this error.
            This represents a traceback of sorts.

    """

    def __init__(self, message, refs=[]):
        """Initialize an InferenceError."""
        super().__init__(message, refs)
        self.message = message
        self.refs = refs
        self.traceback_refs = infer_trace.get()

    def print_tb_end(self, fn_ctx, args_ctx, is_prim):
        """Print the error message at the end of a traceback."""
        eprint(f'{type(self).__qualname__}: {self.message}')


class MyiaTypeError(InferenceError):
    """Type error in a Myia program."""

    def print_tb_end(self, fn_ctx, args_ctx, is_prim):
        """Print the error message at the end of a traceback."""
        if fn_ctx is None:
            super().print_tb_end(fn_ctx, args_ctx, is_prim)
            return
        s = f'{type(self).__qualname__}: `{fn_ctx}` cannot be called with' \
            f' argument types {args_ctx}.'
        if is_prim:
            s += f' Reason: {self.message}'
        eprint(s)


class MyiaShapeError(InferenceError):
    """Shape error in a Myia program."""


def type_error_nargs(ident, expected, got):
    """Return a MyiaTypeError for number of arguments mismatch."""
    return MyiaTypeError(
        f"Wrong number of arguments for '{label(ident)}':"
        f" expected {expected}, got {got}."
    )


def check_nargs(ident, expected, args):
    """Return a MyiaTypeError for number of arguments mismatch."""
    got = len(args)
    if expected is not None and got != expected:
        raise type_error_nargs(ident, expected, got)


class TypeDispatchError(MyiaTypeError):
    """Represents an error in type dispatch for a MetaGraph."""

    def __init__(self, metagraph, types, refs=[]):
        """Initialize a TypeDispatchError."""
        message = f'`{metagraph}` is not defined for argument types {types}'
        super().__init__(message, refs=refs)
        self.metagraph = metagraph
        self.types = types

    def print_tb_end(self, fn_ctx, args_ctx, is_prim):
        """Print the error message at the end of a traceback."""
        eprint(f'{type(self).__qualname__}: {self.message}')
