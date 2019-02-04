
from typing import Tuple
from dataclasses import dataclass
from contextvars import ContextVar

from .. import dtype
from ..debug.label import label
from ..utils import Named, Event, Partializable, eprint

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
NOTVISIBLE = Named('NOTVISIBLE')


#####################
# Function wrappers #
#####################


class Possibilities(frozenset):
    pass


@dataclass(frozen=True)
class Function:
    pass


@dataclass(frozen=True)
class GraphFunction(Function):
    graph: 'Graph'
    context: 'Context'
    tracking_id: object = None


@dataclass(frozen=True)
class PrimitiveFunction(Function):
    prim: 'Primitive'
    tracking_id: object = None


@dataclass(frozen=True)
class MetaGraphFunction(Function):
    metagraph: 'MetaGraph'
    context: 'Context'
    tracking_id: object = None


@dataclass(frozen=True)
class PartialApplication(Function):
    fn: object
    args: object


@dataclass(frozen=True)
class JTransformedFunction(Function):
    fn: object


@dataclass(frozen=True)
class VirtualFunction(Function):
    args: Tuple['AbstractBase']
    output: 'AbstractBase'


@dataclass(frozen=True)
class TypedPrimitive(Function):
    prim: 'Primitive'
    args: Tuple['AbstractBase']
    output: 'AbstractBase'


class DummyFunction(Function):
    pass


#################
# Abstract data #
#################


class AbstractBase:

    def make_key(self):
        raise NotImplementedError()

    def key(self):
        if not hasattr(self, '_key'):
            self._key = self.make_key()
        return self._key

    def __eq__(self, other):
        return type(self) is type(other) \
            and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())


class AbstractValue(AbstractBase):
    def __init__(self, values, count=0):
        self.values = TrackDict(values)
        self.count = count

    def make_key(self):
        return tuple(sorted(self.values.items()))


class AbstractScalar(AbstractValue):
    def __repr__(self):
        contents = [f'{k}={v}' for k, v in self.values.items()
                    if v not in (ABSENT, ANYTHING)]
        return f'S({", ".join(contents)})'


class AbstractType(AbstractValue):
    def __init__(self, typ):
        super().__init__({VALUE: typ})

    def __repr__(self):
        return f'Ty({self.values[VALUE]})'


class AbstractError(AbstractValue):
    def __init__(self, err):
        super().__init__({VALUE: err})

    def __repr__(self):
        return f'E({self.values[VALUE]})'


class AbstractFunction(AbstractValue):
    def __init__(self, *poss, value=None):
        v = Possibilities(poss) if value is None else value
        super().__init__({VALUE: v})

    async def get(self):
        v = self.values[VALUE]
        return (await v if isinstance(v, Pending) else v)

    def get_sync(self):
        return self.values[VALUE]

    def get_unique(self):
        poss = self.values[VALUE]
        if isinstance(poss, Pending):
            raise MyiaTypeError('get_unique invalid because Pending')
        if len(poss) != 1:
            raise MyiaTypeError(f'Expected unique function, not {poss}')
        fn, = poss
        return fn

    def __repr__(self):
        return f'Fn({self.values[VALUE]})'


class AbstractTuple(AbstractValue):
    def __init__(self, elements, values=None):
        super().__init__(values or {})
        self.elements = tuple(elements)

    def make_key(self):
        elms = tuple(e.make_key() for e in self.elements)
        return (super().make_key(), elms)

    def __repr__(self):
        return f'T({", ".join(map(repr, self.elements))})'


class AbstractArray(AbstractValue):
    def __init__(self, element, values):
        super().__init__(values)
        self.element = element

    def make_key(self):
        return (super().make_key(), self.element.make_key())

    def __repr__(self):
        return f'A({self.element}, SHAPE={self.values[SHAPE]})'


class AbstractList(AbstractValue):
    def __init__(self, element, values=None):
        super().__init__(values or {})
        self.element = element

    def make_key(self):
        return (super().make_key(), self.element.make_key())

    def __repr__(self):
        return f'L({self.element})'


class AbstractClass(AbstractValue):
    def __init__(self, tag, attributes, methods, values={}):
        super().__init__(values)
        self.tag = tag
        self.attributes = attributes
        self.methods = methods

    def make_key(self):
        attrs = tuple((k, v.make_key()) for k, v in self.attributes.items())
        return (super().make_key(), self.tag, attrs)

    def __repr__(self):
        elems = [f'{k}={v}' for k, v in self.attributes.items()]
        return f'*{self.tag}({", ".join(elems)})'


class AbstractJTagged(AbstractValue):
    def __init__(self, element):
        super().__init__({})
        self.element = element

    def make_key(self):
        return (super().make_key(), self.element.make_key())

    def __repr__(self):
        return f'J({self.element})'


##########
# Tracks #
##########


class TrackDict(dict):
    pass


class Subtrack:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name

    def clone(self, v, recurse):
        return recurse(v)

    async def async_clone(self, v, recurse):
        return await recurse(v)

    def broaden(self, v, recurse, loop):
        return recurse(v, loop)


class _ValueSubtrack(Subtrack):
    def broaden(self, v, recurse, loop):
        return ANYTHING


class _TypeSubtrack(Subtrack):
    pass


class _ShapeSubtrack(Subtrack):
    pass


VALUE = _ValueSubtrack('VALUE')
TYPE = _TypeSubtrack('TYPE')
SHAPE = _ShapeSubtrack('SHAPE')


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

    def __init__(self, message, refs=[], app=None):
        """Initialize an InferenceError."""
        super().__init__(message, refs)
        self.message = message
        self.refs = refs
        self.traceback_refs = infer_trace.get()
        if app is not None:
            self.traceback_refs[app.context] = app

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


class TypeDispatchError(MyiaTypeError):
    """Represents an error in type dispatch for a MetaGraph."""

    def __init__(self, metagraph, types, refs=[], app=None):
        """Initialize a TypeDispatchError."""
        message = f'`{metagraph}` is not defined for argument types {types}'
        super().__init__(message, refs=refs, app=app)
        self.metagraph = metagraph
        self.types = types

    def print_tb_end(self, fn_ctx, args_ctx, is_prim):
        """Print the error message at the end of a traceback."""
        eprint(f'{type(self).__qualname__}: {self.message}')
