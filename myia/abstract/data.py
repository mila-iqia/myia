"""Data structures to represent data in an abstract way, for inference."""


import re
import prettyprinter as pp
from prettyprinter.prettyprinter import pretty_python_value
from typing import Tuple
from dataclasses import dataclass
from contextvars import ContextVar

from .. import dtype
from ..debug.label import label
from ..utils import Named, Partializable, Interned, Atom, Elements, \
    PossiblyRecursive

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
    args: Tuple['AbstractValue']


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

    args: Tuple['AbstractValue']
    output: 'AbstractValue'


@dataclass(frozen=True)
class TypedPrimitive(Function):
    """Represents a Primitive with an explicitly given type signature.

    Attributes:
        prim: The Primitive
        args: The abstract arguments given to the Primitive
        output: The abstract output

    """

    prim: 'Primitive'
    args: Tuple['AbstractValue']
    output: 'AbstractValue'


class DummyFunction(Function):
    """Represents a function that can't be called."""


#################
# Abstract data #
#################


class AbstractValue(Interned, PossiblyRecursive):
    """Base class for all abstract values.

    Attributes:
        values: A dictionary mapping a Track like VALUE or TYPE
            to a value for that track. Different abstract structures
            may have different tracks, e.g. SHAPE for arrays.

    """

    __cache_eqkey__ = True

    def __init__(self, values):
        """Initialize an AbstractValue."""
        super().__init__()
        self.values = TrackDict(values)

    def __eqkey__(self):
        return Atom(self, tuple(sorted(self.values.items())))

    def __repr__(self):
        return f'{type(self).__qualname__}({format_abstract(self)})'


class AbstractAtom(AbstractValue):
    """Base class for abstract values that are not structures."""


class AbstractScalar(AbstractAtom):
    """Represents a scalar (integer, float, bool, etc.)."""

    def dtype(self):
        """Return the type of this scalar."""
        t = self.values[TYPE]
        if isinstance(t, Pending) and t.done():
            t = t.result()
        return t

    def __pretty__(self, ctx):
        rval = pretty_type(self.values[TYPE])
        v = self.values[VALUE]
        if v is not ANYTHING:
            rval += f' = {v}'
        return rval


class AbstractType(AbstractAtom):
    """Represents a type as a first class value."""

    def __init__(self, typ):
        """Initialize an AbstractType."""
        super().__init__({VALUE: typ})

    def __pretty__(self, ctx):
        t = pretty_type(self.values[VALUE])
        return pretty_join(['Ty(', t, ')'])


class AbstractError(AbstractAtom):
    """This represents some kind of problem in the computation.

    For example, when the specializer tries to specialize a graph that is not
    called anywhere, it won't have the information it needs to do that, so it
    may produce the type AbstractError(DEAD). This may not end up being a real
    problem: dead code won't be called anyway, so it doesn't matter if we can't
    type it. Others may be real problems, e.g. AbstractError(POLY) which
    happens when there are multiple ways to type a graph in a given context.
    """

    def __init__(self, err):
        """Initialize an AbstractError."""
        super().__init__({VALUE: err})

    def __pretty__(self, ctx):
        return f'E({self.values[VALUE]})'


class AbstractExternal(AbstractAtom):
    """Represents a value with an external type, coming from Python."""

    def __init__(self, values):
        """Initialize an AbstractExternal."""
        super().__init__(values)

    def __pretty__(self, ctx):
        rval = pretty_type(self.values[TYPE])
        v = self.values[VALUE]
        if v is not ANYTHING:
            rval += f' = {v}'
        return rval


class AbstractFunction(AbstractAtom):
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

    def __pretty__(self, ctx):
        fns = self.get_sync()
        if isinstance(fns, Possibilities):
            fns = [pretty_python_value(fn, ctx) for fn in fns]
        else:
            fns = [str(fns)]
        return pretty_join(fns, sep=' | ')


class AbstractStructure(AbstractValue):
    """Base class for abstract values that are structures."""

    def __eqkey__(self):
        return Elements(self, super().__eqkey__(), self.children())


class AbstractTuple(AbstractStructure):
    """Represents a tuple of elements."""

    def __init__(self, elements, values=None):
        """Initialize an AbstractTuple."""
        super().__init__(values or {})
        if elements is not ANYTHING:
            elements = tuple(elements)
        self.elements = elements

    def children(self):
        """Return all elements in the tuple."""
        return self.elements

    def __pretty__(self, ctx):
        return pretty_call(ctx, "", self.elements)


class AbstractArray(AbstractStructure):
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

    def children(self):
        """Return the array element."""
        return self.element,

    def __pretty__(self, ctx):
        elem = pretty_python_value(self.element, ctx)
        shp = self.values[SHAPE]
        if isinstance(shp, tuple):
            shp = ['?' if s is ANYTHING else str(s) for s in shp]
        else:
            shp = str(shp)
        shp = pretty_join(shp, ' x ')
        return pretty_join([elem, ' x ', shp])


class AbstractList(AbstractStructure):
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

    def children(self):
        """Return the list element."""
        return self.element,

    def __pretty__(self, ctx):
        elem = pretty_python_value(self.element, ctx)
        return pretty_join(['[', elem, ']'])


class AbstractClass(AbstractStructure):
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

    def children(self):
        """Return the attribute values."""
        return tuple(self.attributes.values())

    def __eqkey__(self):
        vals = AbstractValue.__eqkey__(self)
        return Elements(self, vals, self.tag, self.attributes)

    def __pretty__(self, ctx):
        tagname = self.tag.__qualname__
        return pretty_struct(ctx, tagname, [], self.attributes)


class AbstractJTagged(AbstractStructure):
    """Represents a value (non-function) transformed through J."""

    def __init__(self, element):
        """Initialize an AbstractJTagged."""
        super().__init__({})
        self.element = element

    def children(self):
        """Return the jtagged element."""
        return self.element,

    def __pretty__(self, ctx):
        return pretty_call(ctx, "J", self.element)


class AbstractUnion(AbstractStructure):
    """Represents the union of several possible abstract types."""

    def __init__(self, options):
        """Initialize an AbstractUnion."""
        super().__init__({})
        self.options = frozenset(options)

    def children(self):
        """Return the set of options."""
        return self.options

    def __pretty__(self, ctx):
        return pretty_call(ctx, "U", self.options)


def abstract_union(options):
    """Create a union if necessary.

    If only one opt is given in the options list, return that option.
    """
    opts = []
    for option in options:
        if isinstance(option, AbstractUnion):
            opts += option.options
        else:
            opts.append(option)
    opts = frozenset(opts)
    if len(opts) == 1:
        opt, = opts
        return opt
    else:
        return AbstractUnion(opts)


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

    def __str__(self):  # pragma: no cover
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


class MyiaTypeError(InferenceError):
    """Type error in a Myia program."""


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


class TypeMismatchError(MyiaTypeError):
    """Error to generate when expecting a type and getting another."""

    def __init__(self, expected, got):
        """Initialize a TypeMismatchError."""
        message = f'Expected {expected}, but got {got}'
        super().__init__(message)
        self.expected = expected
        self.got = got


#############################
# Pretty printing utilities #
#############################


def _force_sequence(x):
    if isinstance(x, (list, tuple)):
        return x
    else:
        return [x]


def format_abstract(a):
    """Pretty print an AbstractValue."""
    rval = pp.pformat(a)
    rval = re.sub(r'<<([^>]+)>>=', r'\1', rval)
    return rval


def pretty_type(t):
    """Pretty print a type."""
    if not isinstance(t, type):
        return str(t)
    elif issubclass(t, dtype.Float):
        return f'f{getattr(t, "bits", "??")}'
    elif issubclass(t, dtype.Int):
        return f'i{getattr(t, "bits", "??")}'
    elif issubclass(t, dtype.UInt):
        return f'u{getattr(t, "bits", "??")}'
    else:
        return str(t)


def pretty_call(ctx, title, args, sep=' :: '):
    """Pretty print a call."""
    args = _force_sequence(args)
    return pp.pretty_call_alt(ctx, str(title), args, {})


def pretty_struct(ctx, title, args, kwargs, sep=' :: '):
    """Pretty print a struct."""
    kwargs = {f'{k}<<{sep}>>': v
              for k, v in kwargs.items()}
    return pp.pretty_call_alt(ctx, str(title), args, kwargs)


def pretty_join(elems, sep=None):
    """Join a list of elements."""
    elems = _force_sequence(elems)
    if sep:
        parts = []
        for elem in elems:
            parts += (elem, sep)
        parts = parts[:-1]
    else:
        parts = elems

    return pp.doc.concat(parts)


@pp.register_pretty(AbstractValue)
def _pretty_avalue(a, ctx):
    return a.__pretty__(ctx)
