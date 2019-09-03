"""Data structures to represent data in an abstract way, for inference."""


import inspect
import re
import traceback
from dataclasses import dataclass
from typing import List, Tuple

import prettyprinter as pp
from prettyprinter.prettyprinter import pretty_python_value

from .. import dtype
from ..ir import ANFNode, Constant, Graph, MetaGraph
from ..prim import Primitive
from ..utils import (
    Atom,
    AttrEK,
    Cons,
    Empty,
    InferenceError,
    Interned,
    MyiaTypeError,
    MyiaValueError,
    Named,
    OrderedSet,
    PossiblyRecursive,
    keyword_decorator,
)
from .loop import Pending
from .ref import Context, Reference

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


@dataclass(frozen=True)
class MacroFunction(Function):
    """Represents a Macro.

    Attributes:
        macro: The macro

    """

    macro: 'Macro'


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

    def dtype(self):
        """Return the type of this scalar."""
        t = self.values[TYPE]
        if isinstance(t, Pending) and t.done():
            t = t.result()
        return t

    def __eqkey__(self):
        return Atom(self, tuple(sorted(self.values.items())))

    def __repr__(self):
        return f'{type(self).__qualname__}({format_abstract(self)})'


class AbstractAtom(AbstractValue):
    """Base class for abstract values that are not structures."""


class AbstractScalar(AbstractAtom):
    """Represents a scalar (integer, float, bool, etc.)."""

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

    def __init__(self, elements, values={}):
        """Initialize an AbstractTuple."""
        super().__init__({TYPE: dtype.Tuple, **values})
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


class AbstractDict(AbstractStructure):
    """Represents a dictionary.

    Dictionaries must have the same type for all the values.

    Attributes:
      entries: dict mapping string keys to types

    """

    def __init__(self, entries, values={}):
        """Initalize an AbstractDict."""
        super().__init__({TYPE: dtype.Dict, **values})
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
        constructor: A function to use to build a Python instance.
            Defaults to the tag.

    """

    def __init__(self, tag, attributes, *, values={},
                 constructor=None):
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


class AbstractKeywordArgument(AbstractStructure):
    """Represents a keyword argument."""

    def __init__(self, key, argument):
        """Initialize an AbstractKeywordArgument."""
        super().__init__({})
        self.key = key
        self.argument = argument

    def children(self):
        """Return the argument."""
        return self.argument,

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(self, (v, Atom(self.key, self.key), 'argument'))

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

    def broaden(self, v, recurse, *args):
        """Make a value more generic.

        By default, this amounts to a straight copy.
        """
        return recurse(v, *args)

    def default(self):
        """Return the default value for the track."""
        raise NotImplementedError(f'There is no default for track {self}')


class _ValueTrack(Track):
    """Represents the VALUE track."""

    def broaden(self, v, recurse, *args):
        """Values are broadened to ANYTHING."""
        return ANYTHING


class _TypeTrack(Track):
    """Represents the TYPE track, for scalars."""


class _ShapeTrack(Track):
    """Represents the SHAPE track, for arrays."""


class _AliasIdTrack(Track):
    """Represents the ALIASID track."""

    def merge(self, recurse, v1, v2, forced, bp):
        """Merge two values."""
        # For the time being we don't propagate ALIASID through merge.
        return ABSENT

    def default(self):
        return ABSENT


VALUE = _ValueTrack('VALUE')
TYPE = _TypeTrack('TYPE')
SHAPE = _ShapeTrack('SHAPE')
DATA = _ValueTrack('DATA')
ALIASID = _AliasIdTrack('ALIASID')


##########################
# List-related utilities #
##########################


empty = AbstractADT(Empty, {})


def listof(t):
    """Return the type of a list of t."""
    rval = AbstractADT.new(Cons, {'head': t, 'tail': None})
    rval.attributes['tail'] = AbstractUnion.new([empty, rval])
    return rval.intern()


##########
# Macros #
##########


@dataclass
class MacroInfo:
    """Contains standard information given to macros."""

    engine: object
    outref: object
    argrefs: object
    graph: object
    args: object
    abstracts: object


class Macro:
    """Represents a function that transforms the subgraph it receives."""

    def __init__(self, *, name, infer_args=True):
        """Initialize a Macro."""
        self.name = name
        self.infer_args = infer_args

    async def macro(self, info):
        """Execute the macro proper."""
        raise NotImplementedError(self.name)

    async def reroute(self, engine, outref, argrefs):
        """Reroute a node."""
        if self.infer_args:
            abstracts = [await argref.get() for argref in argrefs]
        else:
            abstracts = None
        info = MacroInfo(
            engine=engine,
            outref=outref,
            argrefs=argrefs,
            graph=outref.node.graph,
            args=[argref.node for argref in argrefs],
            abstracts=abstracts,
        )
        rval = await self.macro(info)
        if isinstance(rval, ANFNode):
            rval = engine.ref(rval, outref.context)
        if not isinstance(rval, Reference):
            raise InferenceError(
                f"Macro '{self.name}' returned {rval}, but it must always "
                f"return an ANFNode (Apply, Constant or Parameter) or a "
                f"Reference."
            )
        return rval

    def __str__(self):
        return f'<Macro {self.name}>'

    __repr__ = __str__


class StandardMacro(Macro):
    """Represents a function that transforms the subgraph it receives."""

    def __init__(self, macro, *, name=None, infer_args=True):
        """Initialize a Macro."""
        super().__init__(name=name or macro.__qualname__,
                         infer_args=infer_args)
        if not inspect.iscoroutinefunction(macro):
            raise TypeError(
                f"Error defining macro '{self.name}':"
                f" macro must be a coroutine defined using async def"
            )
        self._macro = macro

    async def macro(self, info):
        """Execute the macro proper."""
        return await self._macro(info)


@keyword_decorator
def macro(fn, **kwargs):
    """Create a macro out of a function."""
    return StandardMacro(fn, **kwargs)


class MacroError(InferenceError):
    """Wrap an error raised inside a macro."""

    def __init__(self, error):
        """Initialize a MacroError."""
        tb = traceback.format_exception(
            type(error),
            error,
            error.__traceback__,
            limit=7
        )
        del tb[1]
        tb = "".join(tb)
        super().__init__(None, refs=[], pytb=tb)


class MyiaStatic(Macro):
    """Represents a function that can be run at compile time.

    This is simpler, but less powerful than Macro.
    """

    def __init__(self, macro, *, name=None):
        """Initialize a MyiaStatic."""
        super().__init__(name=name or macro.__qualname__,
                         infer_args=True)
        self._macro = macro

    async def macro(self, info):
        """Execute the macro."""
        from .utils import build_value

        def bv(x, ref):
            try:
                return build_value(x)
            except ValueError:
                raise MyiaValueError(
                    'Arguments to a myia_static function must be constant',
                    refs=[ref]
                )
        posargs = []
        kwargs = {}
        for ref, arg in zip(info.argrefs, info.abstracts):
            if isinstance(arg, AbstractKeywordArgument):
                kwargs[arg.key] = bv(arg.argument, ref)
            else:
                posargs.append(bv(arg, ref))
        try:
            rval = self._macro(*posargs, **kwargs)
        except Exception as e:
            raise MacroError(e)
        return Constant(rval)


@keyword_decorator
def myia_static(fn, **kwargs):
    """Create a function that can be run by the inferrer at compile time."""
    return MyiaStatic(fn, **kwargs)


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
