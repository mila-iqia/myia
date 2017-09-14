"""
This file contains essentially everything that's about Symbol
manipulation.

* An assortment of special characters to modify symbols.
* ``GenSym``: generate Symbols.
* ``ParseEnv``: map Symbols to LambdaNode or ValueNode.
"""


from typing import \
    List, Tuple as TupleT, Iterable, Dict, Set, Union, \
    cast, TypeVar, Any
from .nodes import Symbol, LambdaNode
from ..util import EventDispatcher
from ..lib import Pending
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
HIDGLOB = '$'
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
    return Symbol(name, namespace='builtin')


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


class ParseEnv:
    """
    A mapping from Symbol instances to LambdaNode instances. When
    a function is compiled, a ParseEnv will contain all the
    functions created during the compilation. These functions
    will refer to each other with Symbols and the ParseEnv
    connects them to the relevant code.

    A ParseEnv is associated to a GenSym instance that can
    generate fresh Symbols, guaranteed not to already be
    present in the mapping.

    You can index the ParseEnv like a dict:

    >>> parse_env[some_symbol] = some_lambda
    >>> parse_env[some_symbol]
    some_lambda

    When a new Symbol is defined, the ParseEnv also emits the
    ``declare`` event.

    Attributes:
        namespace (str): The namespace in which this ParseEnv's
            Symbols live.
        gen (GenSym): The Symbol generator for this ParseEnv.
        url (str): The filename in which the compiled code
            originally came from.
        bindings ({Symbol: LambdaNode}}): The mapping.
        events (EventDispatcher): Events that this object might
            emit, chiefly the ``declare`` event.

    Events:
        declare(event, symbol, lbda): Triggered when a new mapping
            is added. You can listen to this to track the various
            functions that are being compiled. Use
            ``@parse_env.events.on_declare`` as a decorator.
    """

    def __init__(self,
                 namespace: str = None,
                 gen: 'GenSym' = None,
                 url: str = None) -> None:
        if namespace is None:
            namespace = str(uuid())
        if namespace.startswith('global'):
            self.events = EventDispatcher(self)
        else:
            self.events = None
        self.url = url
        self.gen: GenSym = gen or GenSym(namespace)
        self.bindings: Dict[Symbol, LambdaNode] = {}

    def update(self, bindings) -> None:
        """
        Set several bindings at once.

        Same as ``self.bindings.update(bindings)``
        """
        for k, v in bindings.items():
            self[k] = v

    def __getitem__(self, name) -> LambdaNode:
        return self.bindings[name]

    def __setitem__(self, name, value) -> None:
        self.bindings[name] = value
        if self.events:
            self.events.emit_declare(name, value)


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


globals_pool: Dict[Symbol, Any] = {}
globals_sources: Dict[str, Dict[str, Any]] = {}


def add_source(file, globals):
    print('add_source', file)
    globals_sources[file] = globals


def acquire(sym):
    print('acquire', sym)
    if sym.namespace.startswith('global:'):
        file = sym.namespace[7:]
    else:
        file = ':' + sym.namespace
    globs = globals_sources[file]
    try:
        v = globs[sym.label]
    except KeyError as err:
        builtins = globs['__builtins__']
        if isinstance(builtins, dict):
            # I don't know why this ever happens, but it does.
            v = builtins[sym.label]
        else:
            v = getattr(builtins, sym.label)
    globals_pool = v
    return v


def resolve(sym):
    print('resolve', sym)
    try:
        return globals_pool[sym]
    except KeyError:
        return acquire(sym)


def associate(sym, node):
    if isinstance(node, LambdaNode):
        node.ref = sym
    globals_pool[sym] = node


def create_lambda(ref, args, body, gen=None,
                  global_env=None, globals=None,
                  commit=True, **kw):
    lbda = LambdaNode(args, body, gen, global_env, **kw)
    lbda.globals = globals
    if commit:
        associate(ref, lbda)
    return lbda
