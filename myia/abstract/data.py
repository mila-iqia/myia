"""Data structures to represent data in an abstract way, for inference."""


import re
import prettyprinter as pp
from prettyprinter.prettyprinter import pretty_python_value
from typing import Tuple, List
from dataclasses import dataclass
from contextvars import ContextVar

from ..debug.label import label
from ..utils import Named, Partializable, Interned, Atom, AttrEK, \
    PossiblyRecursive, OrderedSet, dataclass_methods, Cons, Empty
from ..ir import Graph, MetaGraph
from ..prim import Primitive

from .loop import Pending
from .ref import Context


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


class Possibilities(list):
    """Represents a set of possible values.

    This is technically implemented as a list, because the possibility of
    recursive types or values, which may be incomplete when a Possibilities is
    constructed, may impair the equality comparisons needed to construct a set.
    """

    def __init__(self, options):
        """Initialize Possibilities."""
        # We use OrderedSet to trim the options, but it is possible that some
        # of the options which appear unequal at this point will turn out to be
        # equal later on, so we cannot rely on the fact that each option in the
        # list is different.
        super().__init__(OrderedSet(options))

    def __hash__(self):
        return hash(tuple(self))


class TaggedPossibilities(list):
    """Represents a set of possible tag/type combos.

    Each entry in the list must be a list of [tag, type], the tag being an
    integer.
    """

    def __init__(self, options):
        """Initialize TaggedPossibilities."""
        opts = [[tag, typ] for tag, typ in sorted(set(map(tuple, options)))]
        super().__init__(opts)

    def get(self, tag):
        """Get the type associated to the tag."""
        for i, t in self:
            if i == tag:
                return t
        else:
            raise KeyError(tag)

    def __hash__(self):
        return hash(tuple((i, t) for i, t in self))


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

    graph: Graph
    context: Context
    tracking_id: object = None


@dataclass(frozen=True)
class PrimitiveFunction(Function):
    """Represents a Primitive.

    Attributes:
        prim: The primitive
        tracking_id: Identifies different uses of the same primitive.

    """

    prim: Primitive
    tracking_id: object = None


@dataclass(frozen=True)
class MetaGraphFunction(Function):
    """Represents a MetaGraph in a certain Context.

    Attributes:
        metagraph: The metagraph
        context: The context, or Context.empty()
        tracking_id: Identifies different uses of the same metagraph.

    """

    metagraph: MetaGraph
    context: Context
    tracking_id: object = None


@dataclass(eq=False)
class PartialApplication(Function):
    """Represents a partial application.

    Attributes:
        fn: A Function
        args: The first few arguments of that function

    """

    fn: Function
    args: List['AbstractValue']

    def __eqkey__(self):
        return AttrEK(self, ('fn', 'args'))


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

    prim: Primitive
    args: Tuple['AbstractValue']
    output: 'AbstractValue'


@dataclass(frozen=True)
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

    def __init__(self, err, data=None):
        """Initialize an AbstractError."""
        super().__init__({VALUE: err, DATA: data})

    def __pretty__(self, ctx):
        return f'E({self.values[VALUE]})'


class AbstractBottom(AbstractAtom):
    """Represents the type of an expression that does not return."""

    def __init__(self):
        """Initialize an AbstractBottom."""
        super().__init__({})

    def __pretty__(self, ctx):
        return '⊥'


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

    def _maybe_get_unique(self):
        poss = self.values[VALUE]
        if isinstance(poss, Pending):  # pragma: no cover
            # This is a bit circumstantial and difficult to test explicitly
            raise MyiaTypeError('get_unique invalid because Pending')
        poss = frozenset(poss)
        if len(poss) != 1:
            return None
        fn, = poss
        return fn

    def get_unique(self):
        """If there is exactly one possible function, return it.

        Otherwise, raise a MyiaTypeError.
        """
        rval = self._maybe_get_unique()
        if rval is None:
            raise MyiaTypeError(
                f'Expected unique function, not {self.values[VALUE]}'
            )
        return rval

    def get_prim(self):
        """If this AbstractFunction represents a a Primitive, return it.

        Otherwise, return None.
        """
        fn = self._maybe_get_unique()
        if isinstance(fn, (PrimitiveFunction, TypedPrimitive)):
            return fn.prim
        else:
            return None

    def __eqkey__(self):
        return AttrEK(self, ('values',))

    def __pretty__(self, ctx):
        fns = self.get_sync()
        if isinstance(fns, Possibilities):
            fns = [pretty_python_value(fn, ctx) for fn in fns]
        else:
            fns = [str(fns)]
        return pretty_join(fns, sep=' | ')


class AbstractStructure(AbstractValue):
    """Base class for abstract values that are structures."""


class AbstractTuple(AbstractStructure):
    """Represents a tuple of elements."""

    def __init__(self, elements, values=None):
        """Initialize an AbstractTuple."""
        super().__init__(values or {})
        if elements is not ANYTHING:
            elements = list(elements)
        self.elements = elements

    def children(self):
        """Return all elements in the tuple."""
        return self.elements

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(self, (v, 'elements'))

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

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(self, (v, 'element'))

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

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(self, (v, 'element'))

    def __pretty__(self, ctx):
        elem = pretty_python_value(self.element, ctx)
        return pretty_join(['[', elem, ']'])


class AbstractDict(AbstractStructure):
    """Represents a dictionary.

    Dictionaries must have the same type for all the values.

    Attributes:
      entries: dict mapping string keys to types

    """

    def __init__(self, entries, values=None):
        """Initalize an AbstractDict."""
        super().__init__(values or {})
        self.entries = entries

    def children(self):
        """Return the dict element."""
        return tuple(self.entries.values())

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(self, (v, 'entries',
                             Atom(self, tuple(self.entries.keys()))))

    def __pretty__(self, ctx):
        lst = ['{']
        for k, v in self.entries.items():
            lst.append(f'{k}: {pretty_python_value(v, ctx)}')
        lst.append('}')
        return pretty_join(lst, sep=',\n')


class AbstractClassBase(AbstractStructure):
    """Represents a class with named attributes and methods.

    Attributes:
        tag: A pointer to the original Python class
        attributes: Maps each field name to a corresponding AbstractValue.
        methods: Maps method names to corresponding functions, which will
            be parsed and converted by the engine when necessary, with the
            instance as the first argument.

    """

    def __init__(self, tag, attributes, methods, values={}, *,
                 constructor=None):
        """Initialize an AbstractClass."""
        super().__init__(values)
        self.tag = tag
        self.attributes = attributes
        self.methods = methods
        if constructor is None:
            constructor = tag
        self.constructor = constructor

    def user_defined_version(self):
        """Return the user-defined version of this type.

        This uses the attribute types as defined by the user, rather than what
        is generated by the inferrer or other methods.
        """
        from .utils import type_to_abstract
        return type_to_abstract(self.tag)

    def children(self):
        """Return the attribute values."""
        return tuple(self.attributes.values())

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(self, (v, 'tag', 'attributes'))

    def __pretty__(self, ctx):
        tagname = self.tag.__qualname__
        return pretty_struct(ctx, tagname, [], self.attributes)


class AbstractClass(AbstractClassBase):
    """Represents a class, typically those defined using @dataclass."""


class AbstractADT(AbstractClassBase):
    """Represents an algebraic data type.

    Unlike AbstractClass, this is suitable to define recursive types. Nested
    ADTs with the same tag must have the same attribute types, which is
    enforced with the normalize_adt function.
    """


class AbstractJTagged(AbstractStructure):
    """Represents a value (non-function) transformed through J."""

    def __init__(self, element):
        """Initialize an AbstractJTagged."""
        super().__init__({})
        self.element = element

    def children(self):
        """Return the jtagged element."""
        return self.element,

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(self, (v, 'element'))

    def __pretty__(self, ctx):
        return pretty_call(ctx, "J", self.element)


class AbstractUnion(AbstractStructure):
    """Represents the union of several possible abstract types.

    Attributes:
        options: A list of possible types. Technically, this should be
            understood as a set, but there are a few issues with using
            sets in the context of recursive types, chiefly the fact that
            an AbstractUnion could be constructed with types that are
            currently incomplete and therefore cannot be compared for
            equality.

    """

    def __init__(self, options):
        """Initialize an AbstractUnion."""
        super().__init__({})
        if isinstance(options, Pending):
            self.options = options
        else:
            self.options = Possibilities(options)

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(self, (v, 'options'))

    def __pretty__(self, ctx):
        return pretty_call(ctx, "U", self.options)


class AbstractTaggedUnion(AbstractStructure):
    """Represents a tagged union.

    Attributes:
        options: A list of (tag, type) pairs.

    """

    def __init__(self, options):
        """Initialize an AbstractTaggedUnion."""
        super().__init__({})
        if isinstance(options, Pending):
            self.options = options
        else:
            self.options = TaggedPossibilities(options)

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(self, (v, 'options'))

    def __pretty__(self, ctx):
        return pretty_struct(ctx, 'U', [], dict(self.options))


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

    def broaden(self, v, recurse, *args):
        """Make a value more generic.

        By default, this amounts to a straight copy.
        """
        return recurse(v, *args)


class _ValueTrack(Track):
    """Represents the VALUE track."""

    def broaden(self, v, recurse, *args):
        """Values are broadened to ANYTHING."""
        return ANYTHING


class _TypeTrack(Track):
    """Represents the TYPE track, for scalars."""


class _ShapeTrack(Track):
    """Represents the SHAPE track, for arrays."""


VALUE = _ValueTrack('VALUE')
TYPE = _TypeTrack('TYPE')
SHAPE = _ShapeTrack('SHAPE')
DATA = _ValueTrack('DATA')


##########
# Errors #
##########


infer_trace = ContextVar('infer_trace')
infer_trace.set({})


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


##########################
# List-related utilities #
##########################


empty = AbstractADT(Empty, {}, dataclass_methods(Empty))


def listof(t):
    rval = AbstractADT.new(Cons, {'head': t, 'tail': None},
                           dataclass_methods(Cons))
    rval.attributes['tail'] = AbstractUnion.new([empty, rval])
    return rval.intern()


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
    try:
        if getattr(a, '_incomplete', False):  # pragma: no cover
            return f'<{a.__class__.__qualname__}: incomplete>'
        else:
            return a.__pretty__(ctx)
    except Exception:  # pragma: no cover
        # Pytest fails badly without this failsafe.
        return f'<{a.__class__.__qualname__}: error in printing>'
