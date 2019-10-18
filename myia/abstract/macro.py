"""Base classes for macros."""

import inspect
import traceback
from collections import defaultdict
from dataclasses import dataclass

from .. import xtype
from ..ir import ANFNode, Constant
from ..utils import (
    InferenceError,
    MyiaTypeError,
    MyiaValueError,
    check_nargs,
    keyword_decorator,
)
from .data import AbstractKeywordArgument, AbstractValue
from .loop import force_pending
from .ref import Reference


class AnnotationBasedChecker:
    """Utility class to check args based on a function signature."""

    def __init__(self, name, fn, nstdargs, allow_varargs=True):
        """Initialize an AnnotationBasedChecker."""
        self.name = name
        data = inspect.getfullargspec(fn)
        if (data.varkw is not None
                or data.defaults is not None
                or data.kwonlyargs
                or data.kwonlydefaults is not None
                or data.varargs and not allow_varargs):
            raise TypeError(
                f'Function {fn} must only have positional arguments'
                f' and no defaults.'
            )
        self.data = data
        self.nargs = None if data.varargs else len(data.args) - nstdargs
        self.typemap = defaultdict(list)
        for i, arg in enumerate(data.args):
            if arg in data.annotations:
                self.typemap[data.annotations[arg]].append(i - nstdargs)
        self.outtype = data.annotations.get('return', None)

    async def check(self, engine, argrefs, uniform=False):
        """Check that the argrefs match the function signature."""
        check_nargs(self.name, self.nargs, argrefs)
        outtype = self.outtype

        async def _force_abstract(x):
            return (await x.get()) if isinstance(x, Reference) else x

        for typ, indexes in self.typemap.items():
            args = [await _force_abstract(argrefs[i]) for i in indexes]

            if uniform:
                res = engine.check(typ, *args)
                if typ == self.outtype:
                    outtype = res
                continue

            for arg in args:
                if isinstance(typ, xtype.TypeMeta):
                    await force_pending(engine.check(typ, arg.xtype(), typ))
                elif isinstance(typ, type) and issubclass(typ, AbstractValue):
                    if not isinstance(arg, typ):
                        raise MyiaTypeError(
                            f'Wrong type {arg} != {typ} for {self.name}'
                        )
                elif callable(typ):
                    await force_pending(engine.check(typ, arg))
                else:
                    raise AssertionError(f'Invalid annotation: {typ}')

        return outtype


@dataclass
class MacroInfo:
    """Contains standard information given to macros."""

    engine: object
    outref: object
    argrefs: object
    graph: object

    async def abstracts(self):
        """Return all the abstract values for the arguments."""
        return [await ref.get() for ref in self.argrefs]

    def nodes(self):
        """Return all the graph nodes for the arguments."""
        return [ref.node for ref in self.argrefs]

    async def build_all(self, *refs):
        """Get constant values from a list of references."""
        return [await self.build(ref) for ref in refs]

    async def build(self, ref, ab=None):
        """Get a constant value from a reference."""
        from .utils import build_value
        if ab is None:
            ab = await ref.get()
        try:
            return build_value(ab)
        except ValueError:
            raise MyiaValueError(
                'Arguments to a myia_static function must be constant',
                refs=[ref]
            )


class Macro:
    """Represents a function that transforms the subgraph it receives."""

    def __init__(self, *, name):
        """Initialize a Macro."""
        self.name = name

    async def macro(self, info):
        """Execute the macro proper."""
        raise NotImplementedError(self.name)

    async def reroute(self, engine, outref, argrefs):
        """Reroute a node."""
        info = MacroInfo(
            engine=engine,
            outref=outref,
            argrefs=argrefs,
            graph=outref.node.graph,
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

    def __init__(self, macro, *, name=None):
        """Initialize a Macro."""
        super().__init__(name=name or macro.__qualname__)
        if not inspect.iscoroutinefunction(macro):
            raise TypeError(
                f"Error defining macro '{self.name}':"
                f" macro must be a coroutine defined using async def"
            )
        self.checker = AnnotationBasedChecker(self.name, macro, 1)
        self._macro = macro

    async def macro(self, info):
        """Execute the macro proper."""
        await self.checker.check(info.engine, info.argrefs)
        return await self._macro(info, *info.argrefs)


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
        super().__init__(name=name or macro.__qualname__)
        self._macro = macro

    def __call__(self, *args):
        """Call self.macro via __call__."""
        return self._macro(*args)

    async def macro(self, info):
        """Execute the macro."""
        posargs = []
        kwargs = {}
        for ref, arg in zip(info.argrefs, await info.abstracts()):
            if isinstance(arg, AbstractKeywordArgument):
                kwargs[arg.key] = await info.build(ref, ab=arg.argument)
            else:
                posargs.append(await info.build(ref, ab=arg))
        try:
            rval = self._macro(*posargs, **kwargs)
        except InferenceError as e:
            raise
        except Exception as e:
            raise MacroError(e)
        return Constant(rval)


@keyword_decorator
def myia_static(fn, **kwargs):
    """Create a function that can be run by the inferrer at compile time."""
    return MyiaStatic(fn, **kwargs)


__all__ = [
    'AnnotationBasedChecker',
    'Macro',
    'MacroError',
    'MacroInfo',
    'MyiaStatic',
    'StandardMacro',
    'macro',
    'myia_static',
]
