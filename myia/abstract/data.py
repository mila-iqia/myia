"""Data structures to represent data in an abstract way, for inference."""

import re
from dataclasses import dataclass
from typing import List, Tuple

import prettyprinter as pp
from prettyprinter.prettyprinter import pretty_python_value

from .. import xtype
from ..classes import Cons, Empty
from ..ir import Graph, MetaGraph
from ..operations import Primitive
from ..utils import (
    Atom,
    AttrEK,
    Interned,
    MyiaTypeError,
    Named,
    OrderedSet,
    PossiblyRecursive,
)
from .loop import Pending
from .ref import Context

# Represents the absence of inferred data
ABSENT = Named("ABSENT")

# Represents an unknown value
ANYTHING = Named("ANYTHING")

# Represents inference problems
VOID = Named("VOID")

# Represents specialization problems
DEAD = Named("DEAD")
POLY = Named("POLY")


# Represent a dict description with known value type but unknown keys.
class DictDesc:
    """Dictionary descriptor.

    Helper class to describe dict entries when keys are unknown.
    This will be used for AbstractDict entries when inferring
    it from a type description (e.g. `Dict[str, int]`).
    """

    __slots__ = ("value_type",)

    def __init__(self, value_type):
        """Init dict descriptor with an abstract value.

        value_type must be an AbstractValue or derived.
        """
        self.value_type = value_type

    def to_dict(self, keys):
        """Convert to a real dict with given keys.

        Each key will be associated to the abstract value in output dict.
        """
        return {key: self.value_type for key in keys}

    def keys(self):
        """Return an empty iterable.

        Placeholder to make AbstractDict work correctly, as it is called
        in AbstractDict.__eqkey__.
        """
        return ()


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


@dataclass(frozen=True)
class MacroFunction(Function):
    """Represents a Macro.

    Attributes:
        macro: The macro

    """

    macro: object


@dataclass(eq=False)
class PartialApplication(Function):
    """Represents a partial application.

    Attributes:
        fn: A Function
        args: The first few arguments of that function

    """

    fn: Function
    args: List["AbstractValue"]

    def __eqkey__(self):
        return AttrEK(self, ("fn", "args"))


@dataclass(frozen=True)
class TransformedFunction(Function):
    """Represents a Function processed through some transform.

    Attributes:
        fn: A Function
        transform: The applied transform

    """

    fn: object
    transform: object


@dataclass(frozen=True)
class TypedPrimitive(Function):
    """Represents a Primitive with an explicitly given type signature.

    Attributes:
        prim: The Primitive
        args: The abstract arguments given to the Primitive
        output: The abstract output

    """

    prim: Primitive
    args: Tuple["AbstractValue"]
    output: "AbstractValue"


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

    def xtype(self):
        """Return the type of this AbstractValue."""
        t = self.values.get(TYPE, None)
        if isinstance(t, Pending) and t.done():
            t = t.result()
        return t

    def xvalue(self):
        """Return the value of this AbstractValue."""
        return self.values[VALUE]

    def __eqkey__(self):
        return Atom(self, tuple(sorted(self.values.items())))

    def __repr__(self):
        return f"{type(self).__qualname__}({format_abstract(self)})"


class AbstractAtom(AbstractValue):
    """Base class for abstract values that are not structures."""


class AbstractScalar(AbstractAtom):
    """Represents a scalar (integer, float, bool, etc.)."""

    def __pretty__(self, ctx):
        rval = pretty_type(self.values[TYPE])
        v = self.values[VALUE]
        if v is not ANYTHING:
            rval += f" = {v}"
        return rval


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
        return f"E({self.values[VALUE]})"


class AbstractBottom(AbstractAtom):
    """Represents the type of an expression that does not return."""

    def __init__(self):
        """Initialize an AbstractBottom."""
        super().__init__({})

    def __pretty__(self, ctx):
        return "⊥"


class AbstractExternal(AbstractAtom):
    """Represents a value with an external type, coming from Python."""

    def __init__(self, values):
        """Initialize an AbstractExternal."""
        super().__init__(values)

    def __pretty__(self, ctx):
        rval = pretty_type(self.values[TYPE])
        v = self.values[VALUE]
        if v is not ANYTHING:
            rval += f" = {v}"
        return rval


class AbstractFunctionBase(AbstractAtom):
    """Base for function types."""

    def get_prim(self):
        """If this AbstractFunction represents a a Primitive, return it.

        Otherwise, return None.
        """
        return None


class AbstractFunction(AbstractFunctionBase):
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
        return await v if isinstance(v, Pending) else v

    def get_sync(self):
        """Return a set of all possible Functions (synchronous)."""
        return self.values[VALUE]

    def _maybe_get_unique(self):
        poss = self.values[VALUE]
        if isinstance(poss, Pending):  # pragma: no cover
            # This is a bit circumstantial and difficult to test explicitly
            raise MyiaTypeError("get_unique invalid because Pending")
        poss = frozenset(poss)
        if len(poss) != 1:
            return None
        (fn,) = poss
        return fn

    def get_unique(self):
        """If there is exactly one possible function, return it.

        Otherwise, raise a MyiaTypeError.
        """
        rval = self._maybe_get_unique()
        if rval is None:
            raise MyiaTypeError(
                f"Expected unique function, not {self.values[VALUE]}"
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
        return AttrEK(self, ("values",))

    def __pretty__(self, ctx):
        fns = self.get_sync()
        if isinstance(fns, Possibilities):
            fns = [pretty_python_value(fn, ctx) for fn in fns]
        else:
            fns = [str(fns)]
        return pretty_join(fns, sep=" | ")


class AbstractFunctionUnique(AbstractFunctionBase):
    """Represents some function with an explicitly given type signature.

    Unlike AbstractFunction, this represents the type of a single function
    rather than a set of possible functions.

    Attributes:
        args: The abstract arguments given to the function
        output: The abstract output
    """

    def __init__(self, args, output, values={}):
        """Initialize the AbstractFunctionUnique."""
        super().__init__(values)
        self.args = list(args)
        self.output = output

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(self, (v, "args", "output"))

    def __pretty__(self, ctx):
        return pretty_call(ctx, "Fn", (self.args, self.output))


class AbstractRandomState(AbstractAtom):
    """Abstract class to represent a backend random state object."""

    def __init__(self):
        """Initialize an AbstractRandomState."""
        super().__init__({})


class AbstractStructure(AbstractValue):
    """Base class for abstract values that are structures."""


class AbstractWrapper(AbstractStructure):
    """Base class for abstract values that wrap a single element type."""

    def __init__(self, element, values):
        """Initialize an AbstractWrapper."""
        super().__init__(values)
        self.element = element

    def children(self):
        """Return the element."""
        return (self.element,)

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(self, (v, "element"))


class AbstractType(AbstractWrapper):
    """Represents a type as a first class value."""

    def __init__(self, typ, values={}):
        """Initialize an AbstractType."""
        super().__init__(typ, values)

    def __pretty__(self, ctx):
        t = pretty_type(self.element)
        return pretty_join(["Ty(", t, ")"])


class AbstractTuple(AbstractStructure):
    """Represents a tuple of elements."""

    def __init__(self, elements, values={}):
        """Initialize an AbstractTuple."""
        super().__init__({TYPE: xtype.Tuple, **values})
        if elements is not ANYTHING:
            elements = list(elements)
        self.elements = elements

    def children(self):
        """Return all elements in the tuple."""
        return self.elements

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(self, (v, "elements"))

    def __pretty__(self, ctx):
        return pretty_call(ctx, "", self.elements)


class AbstractArray(AbstractWrapper):
    """Represents an array.

    The SHAPE track on an array contains the array's shape.

    Arrays must be homogeneous, hence a single AbstractValue is used to
    represent every element.

    Attributes:
        element: AbstractValue representing each element of the array.

    """

    def xshape(self):
        """Return the shape of this array."""
        return self.values[SHAPE]

    def __pretty__(self, ctx):
        elem = pretty_python_value(self.element, ctx)
        shp = self.values[SHAPE]
        if isinstance(shp, tuple):
            shp = ["?" if s is ANYTHING else str(s) for s in shp]
        else:
            shp = str(shp)
        shp = pretty_join(shp, " x ")
        return pretty_join([elem, " x ", shp])


class AbstractDict(AbstractStructure):
    """Represents a dictionary.

    Dictionaries must have the same type for all the values.

    Attributes:
      entries: dict mapping string keys to types

    """

    def __init__(self, entries, values={}):
        """Initalize an AbstractDict."""
        super().__init__({TYPE: xtype.Dict, **values})
        self.entries = entries

    def children(self):
        """Return the dict element."""
        return tuple(self.entries.values())

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(
            self, (v, "entries", Atom(self, tuple(self.entries.keys())))
        )

    def __pretty__(self, ctx):
        lst = ["{"]
        for k, v in self.entries.items():
            lst.append(f"{k}: {pretty_python_value(v, ctx)}")
        lst.append("}")
        return pretty_join(lst, sep=",\n")


class AbstractClassBase(AbstractStructure):
    """Represents a class with named attributes and methods.

    Attributes:
        tag: A pointer to the original Python class
        attributes: Maps each field name to a corresponding AbstractValue.
        constructor: A function to use to build a Python instance.
            Defaults to the tag.

    """

    def __init__(self, tag, attributes, *, values={}, constructor=None):
        """Initialize an AbstractClass."""
        super().__init__({TYPE: tag, **values})
        self.tag = tag
        self.attributes = attributes
        if constructor is None:
            constructor = tag
        self.constructor = constructor

    def user_defined_version(self):
        """Return the user-defined version of this type.

        This uses the attribute types as defined by the user, rather than what
        is generated by the inferrer or other methods.
        """
        from .to_abstract import type_to_abstract

        return type_to_abstract(self.tag)

    def children(self):
        """Return the attribute values."""
        return tuple(self.attributes.values())

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(self, (v, "tag", "attributes"))

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


class AbstractJTagged(AbstractWrapper):
    """Represents a value (non-function) transformed through J."""

    def __init__(self, element, values={}):
        """Initialize an AbstractJTagged."""
        super().__init__(element, values)

    def __pretty__(self, ctx):
        return pretty_call(ctx, "J", self.element)


class AbstractHandle(AbstractWrapper):
    """Represents a value (non-function) transformed through J."""

    def __init__(self, element, values={}):
        """Initialize an AbstractHandle."""
        super().__init__(element, values)

    def __pretty__(self, ctx):
        return pretty_call(ctx, "H", self.element)


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
        return AttrEK(self, (v, "options"))

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
        return AttrEK(self, (v, "options"))

    def __pretty__(self, ctx):
        return pretty_struct(ctx, "U", [], dict(self.options))


class AbstractKeywordArgument(AbstractStructure):
    """Represents a keyword argument."""

    def __init__(self, key, argument):
        """Initialize an AbstractKeywordArgument."""
        super().__init__({})
        self.key = key
        self.argument = argument

    def children(self):
        """Return the argument."""
        return (self.argument,)

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(self, (v, Atom(self.key, self.key), "argument"))

    def __pretty__(self, ctx):
        return pretty_struct(ctx, "KW", [], {self.key: self.argument})


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

    __repr__ = __str__

    def __lt__(self, other):
        return self.name < other.name

    def merge(self, recurse, v1, v2, forced, bp):
        """Merge two values."""
        return recurse(v1, v2, forced, bp)

    def clone(self, v, recurse):
        """Clone the value associated to this Track in a TrackDict."""
        return recurse(v)

    def broaden(self, v, recurse, **kwargs):
        """Make a value more generic.

        By default, this amounts to a straight copy.
        """
        return recurse(v, **kwargs)

    def default(self):
        """Return the default value for the track."""
        raise NotImplementedError(f"There is no default for track {self}")


class _ValueTrack(Track):
    """Represents the VALUE track."""

    def broaden(self, v, recurse, **kwargs):
        """Values are broadened to ANYTHING."""
        return ANYTHING


class _TypeTrack(Track):
    """Represents the TYPE track, for scalars."""

    def default(self):
        return ABSENT


class _ShapeTrack(Track):
    """Represents the SHAPE track, for arrays."""

    def default(self):
        return ABSENT


class _AliasIdTrack(Track):
    """Represents the ALIASID track."""

    def merge(self, recurse, v1, v2, forced, bp):
        """Merge two values."""
        # For the time being we don't propagate ALIASID through merge.
        return ABSENT

    def default(self):
        return ABSENT


VALUE = _ValueTrack("VALUE")
TYPE = _TypeTrack("TYPE")
SHAPE = _ShapeTrack("SHAPE")
DATA = _ValueTrack("DATA")
ALIASID = _AliasIdTrack("ALIASID")


##########################
# List-related utilities #
##########################


empty = AbstractADT(Empty, {})


def listof(t):
    """Return the type of a list of t."""
    rval = AbstractADT.new(Cons, {"head": t, "tail": None})
    rval.attributes["tail"] = AbstractUnion.new([empty, rval])
    return rval.intern()


###################
# Other utilities #
###################


def u64tup_typecheck(engine, tup):
    """Verify that tup is a tuple of uint64."""
    tup_t = engine.check(AbstractTuple, tup)
    for elem_t in tup_t.elements:
        engine.abstract_merge(xtype.UInt[64], elem_t.xtype())
    return tup_t


def u64pair_typecheck(engine, shp):
    """Verify that tup is a pair of uint64."""
    tup_t = u64tup_typecheck(engine, shp)
    if len(tup_t.elements) != 2:
        raise MyiaTypeError(
            f"Expected Tuple Length 2, not Tuple Length"
            f"{len(tup_t.elements)}"
        )
    return tup_t


def i64tup_typecheck(engine, tup):
    """Verify that tup is a tuple of int64."""
    tup_t = engine.check(AbstractTuple, tup)
    for elem_t in tup_t.elements:
        engine.abstract_merge(xtype.Int[64], elem_t.xtype())
    return tup_t


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
    rval = re.sub(r"<<([^>]+)>>=", r"\1", rval)
    return rval


def pretty_type(t):
    """Pretty print a type."""
    return str(t)


def pretty_call(ctx, title, args, sep=" :: "):
    """Pretty print a call."""
    args = _force_sequence(args)
    return pp.pretty_call_alt(ctx, str(title), args, {})


def pretty_struct(ctx, title, args, kwargs, sep=" :: "):
    """Pretty print a struct."""
    kwargs = {f"{k}<<{sep}>>": v for k, v in kwargs.items()}
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
        if getattr(a, "_incomplete", False):  # pragma: no cover
            return f"<{a.__class__.__qualname__}: incomplete>"
        else:
            return a.__pretty__(ctx)
    except Exception:  # pragma: no cover
        # Pytest fails badly without this failsafe.
        return f"<{a.__class__.__qualname__}: error in printing>"


__all__ = [
    "ABSENT",
    "ALIASID",
    "ANYTHING",
    "DATA",
    "DEAD",
    "POLY",
    "SHAPE",
    "TYPE",
    "VALUE",
    "VOID",
    "AbstractADT",
    "AbstractArray",
    "AbstractAtom",
    "AbstractBottom",
    "AbstractClass",
    "AbstractClassBase",
    "AbstractDict",
    "AbstractError",
    "AbstractExternal",
    "AbstractFunction",
    "AbstractFunctionBase",
    "AbstractHandle",
    "AbstractJTagged",
    "AbstractKeywordArgument",
    "AbstractRandomState",
    "AbstractScalar",
    "AbstractStructure",
    "AbstractTaggedUnion",
    "AbstractTuple",
    "AbstractType",
    "AbstractUnion",
    "AbstractValue",
    "Function",
    "GraphFunction",
    "MacroFunction",
    "MetaGraphFunction",
    "PartialApplication",
    "Possibilities",
    "PrimitiveFunction",
    "TaggedPossibilities",
    "Track",
    "TrackDict",
    "TransformedFunction",
    "TypedPrimitive",
    "AbstractFunctionUnique",
    "empty",
    "format_abstract",
    "i64tup_typecheck",
    "listof",
    "pretty_call",
    "pretty_join",
    "pretty_struct",
    "pretty_type",
    "u64pair_typecheck",
    "u64tup_typecheck",
]
