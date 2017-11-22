"""
This file contains essentially everything that's about Symbol
manipulation.

* An assortment of special characters to modify symbols.
* ``GenSym``: generate Symbols.
* ``globals_pool``: all Lambda nodes that are associated to a
  global symbol (i.e. all of them) are mapped in that dictionary,
  among other things.
* ``create_lambda``: create a new Lambda and register it in the
  globals_pool.
"""


from typing import \
    List, Tuple as TupleT, Iterable, Dict, Set, Union, \
    cast, TypeVar, Any
from .nodes import Symbol, LambdaNode
from ..util import EventDispatcher
from uuid import uuid4 as uuid


###############################################
# Special characters to modify function names #
###############################################


THEN = '✓'
ELSE = '✗'
WTEST = '⤾'
WLOOP = '⥁'
LBDA = 'λ'

JTAG = '↑'
BPROP = '♦'
BPROP_CLOS = '♢'
SENS = '∇'
NULLSYM = '×'
TMP = '◯'
ANORM = 'α'

TMP_SENS = f'{TMP}{SENS}'
TMP_BPROP = f'{TMP}{BPROP_CLOS}'
TMP_LET = f'{TMP}let'


#####################
# Symbol generation #
#####################


class GenSym:
    """
    Symbol generator. The generator creates unique Symbols in the
    given namespace, assuming that it is the only generator (to ever
    exist) for that namespace.

    If it is given None as its initial namespace, GenSym generates
    a unique namespace with the uuid4 method.

    Attributes:
        namespace: The namespace in which the symbols are generated.
    """

    def __init__(self, namespace: str = None) -> None:
        self.varcounts: Dict[str, int] = {}
        self.namespace: str = namespace or str(uuid())

    def inc_version(self, ref: str) -> int:
        """
        Increment the current version number for the variable
        name given as ref.
        """
        if ref in self.varcounts:
            self.varcounts[ref] += 1
            return self.varcounts[ref]
        else:
            self.varcounts[ref] = 1
            return 1

    def sym(self, name: str, version: int = None) -> Symbol:
        """
        Create a unique Symbol with the given name. If one or more
        Symbols with the same name exists, the new Symbol will have
        a higher version number than any of them.
        """
        return Symbol(
            name,
            namespace=self.namespace,
            version=version or self.inc_version(name),
            relation=None
        )

    def rel(self, orig: Symbol, relation: str, version: int = None) -> Symbol:
        """
        Create a new Symbol that relates to the orig Symbol with
        the given relation. The new Symbol is guaranteed not to
        already exist.
        """
        # Note: str(a) == str(b) if a == b, but it is possible (I think?)
        #  that str(a) == str(b) if a != b. This is okay, it just means that
        #  some symbols may have a higher version than strictly
        #  necessary (note that we could use the same count for all
        #  variables, and everything would still work)
        if not version:
            ref = f'{str(orig)}/{relation}'
            version = self.inc_version(ref)
        return Symbol(
            orig,
            namespace=self.namespace,
            version=version,
            relation=relation
        )

    def dup(self, orig: Symbol) -> Symbol:
        """
        Make a new Symbol that has the same name/relation/etc. as the
        original, but a new version.
        """
        return self(orig.label, orig.relation or None)

    def __call__(self, s: Union[str, Symbol],
                 rel: str = None, *,
                 version: int = None) -> Symbol:
        if isinstance(rel, str):
            assert isinstance(s, Symbol)
            return self.rel(s, rel, version)
        else:
            return self.sym(s, version)


def bsym(name: str) -> Symbol:
    """
    Create a builtin symbol.

    A builtin symbol points to a function for internal use
    which is not meant to be referred to by name by the user.
    Accordingly, it will not be present in the user namespace.

    It is the case that ``bsym(x) == bsym(x)``, because
    builtins are indexed by name only.
    """
    return Symbol(name, namespace='global::builtin')


_ngen = GenSym(namespace='null')


def nsym() -> Symbol:
    """
    Create a null symbol.

    Use as a placeholder in destructuring assignments for
    irrelevant elements.

    It is **not** the case that ``nsym() == nsym()``. Each
    null symbol is different. That might be something to
    fix.
    """
    return _ngen(NULLSYM)


def is_global(sym):
    if not isinstance(sym, Symbol):
        return False
    ns = sym.namespace
    return ns.startswith('global:')


def is_builtin(sym):
    return isinstance(sym, Symbol) and sym.namespace == 'global::builtin'


class VariableTracker:
    """
    Track variable names and map them to Symbol instances.
    """

    def __init__(self, parent: 'VariableTracker' = None) -> None:
        self.parent = parent
        self.bindings: Dict[str, Symbol] = {}

    def get_free(self, name: str, preserve_about: bool = True) \
            -> TupleT[bool, 'Symbol']:
        """
        Return whether the given variable name is a free variable
        or not, and the Symbol associated to the variable.

        A variable is free if it is found in this Tracker's parent,
        or grandparent, etc.
        It is not free if it is found in ``self.bindings``.

        Raise NameError if the variable was not declared.
        """
        if name in self.bindings:
            return (False, self.bindings[name].copy(preserve_about))
        elif self.parent is None:
            raise NameError("Undeclared variable: {}".format(name))
        else:
            return (True, self.parent.get_free(name)[1])

    def get(self, name: str, preserve_about: bool = True) -> Symbol:
        return self.get_free(name, preserve_about)[1]

    def __getitem__(self, name: str) -> Symbol:
        return self.get_free(name)[1]

    def __setitem__(self, name: str, value: Symbol) -> None:
        self.bindings[name] = value


###########################################
# Global pool to resolve global variables #
###########################################


class BackedPool(EventDispatcher):
    """
    Represents a pool of Symbol->value associations, backed by a store
    of namespace->dict associations. In a nutshell, this is used to map
    Symbols from global namespaces (e.g. the namespace `global:file.py`)
    to the values of corresponding global variables.
    """

    def __init__(self) -> None:
        self.events = EventDispatcher()
        self.pool: Dict[Symbol, Any] = {}
        self.sources: Dict[str, Dict[str, Any]] = {}

    def add_source(self, namespace, contents):
        self.sources[namespace] = contents

    def acquire(self, sym):
        """
        Acquire the value corresponding to the given symbol from the
        source dictionary corresponding to the symbol's namespace and
        throw a `KeyError` if there is no such value.

        Most of the time, you want to call `pool.resolve(sym)`.
        """
        globs = self.sources[sym.namespace]
        try:
            v = globs[sym.label]
        except KeyError as err:
            builtins = globs['__builtins__']
            try:
                if isinstance(builtins, dict):
                    # I don't know why this ever happens, but it does.
                    v = builtins[sym.label]
                else:
                    v = getattr(builtins, sym.label)
            except (KeyError, AttributeError):
                raise NameError(f"Could not resolve global: '{sym}' "
                                f"in namespace: '{sym.namespace}'.")
        self.events.emit_acquire(sym, v)
        self.pool[sym] = v
        return v

    def resolve(self, sym):
        """
        Returns the value associated with the symbol. In a nutshell:

        >>> resolve(Symbol('x', namespace='global:file.py'))
        <global variable x from file.py>

        As an exception, any function processed by Myia will be associated
        to its LambdaNode in this dictionary (even though it is a FunctionImpl
        instance as seen from Python's globals), and some additional entries
        (gradients, etc.) are present here that are not in Python's globals
        (these entries' Symbols are engineered so they cannot clash with
        anything else).
        """
        try:
            return self.pool[sym]
        except KeyError:
            return self.acquire(sym)

    def associate(self, sym, node):
        """
        Associate the given symbol to the given node (typically a LambdaNode)
        in this pool. The LambdaNode's `ref` field will be set to the
        symbol.
        """
        if isinstance(node, LambdaNode):
            node.ref = sym
        self.events.emit_acquire(sym, node)
        self.pool[sym] = node

    def __getitem__(self, sym):
        return self.resolve(sym)


# Maps global Symbols to whatever it is they resolve to. When compiling
# a function, its LambdaNodes will be registered in there directly.
# Values coming from Python globals are registered there when they are
# requested. Globals from all global namespaces are pooled together,
# which is fine because namespaces avoid clashes.
globals_pool: BackedPool = BackedPool()


def create_lambda(ref, args, body, gen=None, *,
                  commit=True, **kw):
    """
    Create a LambdaNode named according to the `ref` Symbol and return it.
    If `commit` is `True`, `ref` will be associated to the `LambdaNode` in
    `globals_pool`.
    """
    lbda = LambdaNode(args, body, gen, **kw)
    if commit:
        globals_pool.associate(ref, lbda)
    return lbda
