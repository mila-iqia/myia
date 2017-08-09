from typing import \
    List, Tuple as TupleT, Iterable, Dict, Set, Union, \
    cast, TypeVar

from uuid import uuid4 as uuid
from copy import copy
import textwrap
import traceback
from .buche import HReprBase
from .event import EventDispatcher


__save_trace__ = False


Locatable = Union['MyiaASTNode', 'Location', None]


class Location:
    """
    Represents a source code location for an AST node.

    Attributes:
        url (str): The path of the code file.
        line (int): The line number in that file.
        column (int): The column number in that file.
    """

    def __init__(self, url: str, line: int, column: int) -> None:
        self.url = url
        self.line = line
        self.column = column

    def __str__(self) -> str:
        return '{}@{}:{}'.format(self.url, self.line, self.column)

    def traceback(self) -> str:
        """
        Print out a "traceback" that corresponds to this location,
        with the line printed out and a caret at the right column.
        Basically:

        >>> loc.traceback()
          File {url}, line {line}, column {column}
            x = f(y)
                ^

        This is mostly meant for printing out ``MyiaSyntaxError``s.
        """
        try:
            with open(self.url) as file:
                raw_code = file.readlines()[self.line - 1].rstrip("\n")
                raw_caret = ' ' * self.column + '^'
                code, caret = textwrap.dedent(
                    raw_code + '\n' + raw_caret
                ).split("\n")
            return '  File "{}", line {}, column {}\n    {}\n    {}'.format(
                self.url, self.line, self.column, code, caret)
        except FileNotFoundError:
            return '  File "{}", line {}, column {}'.format(
                self.url, self.line, self.column)


def _get_location(x: Locatable) -> Location:
    """
    A helper function that returns a Location corresponding to
    whatever the argument is (a MyiaASTNode or a Location).
    """
    if isinstance(x, MyiaASTNode):
        return x.location
    elif isinstance(x, Location) or x is None:
        return x
    else:
        raise TypeError(f'{x} is not a/has no location')


class ParseEnv:
    """
    A mapping from Symbol instances to Lambda instances. When
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
        bindings ({Symbol: Lambda}}): The mapping.
        events (EventDispatcher): Events that this object might
            emit, chiefly the ``declare`` event.

    Events:
        declare(event, symbol, lbda): Triggered when a new mapping
            is added. You can listen to this to track the various
            functions that are being compiled. Use
            ``@parse_env.events.on_declare`` decorator.
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
        self.bindings: Dict[Symbol, Lambda] = {}

    def update(self, bindings) -> None:
        """
        Set several bindings at once.

        Same as ``self.bindings.update(bindings)``
        """
        for k, v in bindings.items():
            self[k] = v

    def __getitem__(self, name) -> 'Lambda':
        return self.bindings[name]

    def __setitem__(self, name, value) -> None:
        self.bindings[name] = value
        if self.events:
            self.events.emit_declare(name, value)


T = TypeVar('T', bound='MyiaASTNode')


class MyiaASTNode(HReprBase):
    """
    Base class for Myia's AST nodes. This does a few bookkeeping
    operations:

    * If the ``__save_trace__`` global is set, it will save the
      current trace in the node.
    * It initializes a set of annotations.
    * It generates boilerplate HTML for use with ``hrepr``.
    """

    def __init__(self, location: Locatable = None) -> None:
        self.location = _get_location(location)
        if __save_trace__:
            # TODO: make sure that we're always removing the right
            # number of entries
            self.trace = traceback.extract_stack()[:-2]
        else:
            self.trace = None
        self.annotations: Set[str] = set()

    def at(self: T, location: Locatable) -> T:
        """
        Returns a copy of the node but which has the same location
        as the node given as an argument (a new Location can also
        be given directly). This does not modify the current node.
        """
        rval = copy(self)
        rval.location = _get_location(location)
        return rval

    def children(self) -> List['MyiaASTNode']:
        return []

    def __repr__(self) -> str:
        return str(self)

    def __hrepr__(self, H, hrepr):
        rval = H.div[self.__class__.__name__][f'pyid-{id(self)}']
        if self.annotations:
            rval = rval.__getitem__(tuple(self.annotations))
        return rval


class Symbol(MyiaASTNode):
    """
    Represent a variable name in Myia's frontend AST.

    Symbols should not be created directly. They should be created
    through a GenSym factory: GenSym enforces a unique namespace and
    keeps track of versions to guarantee that no Symbols accidentally
    collide.

    Attributes:
        label (str or Symbol): the name of the variable. If
            relation is None, this must be a string, otherwise
            this must be a Symbol.
        namespace (str): the namespace in which the variable
            lives. This is usually 'global', 'builtin', or a
            uuid created on a per-Lambda expression basis.
        version (int): differentiates variables with the same
            name and namespace. This can happen when there are
            multiple writes to the same variable in Python.
        relation (str): how this variable relates to some other
            variable in the 'label' attribute. For example,
            automatic differentiation will accumulate the gradient
            for variable x in a Symbol with label x and relation
            'sensitivity'.

    The HTML pretty-printer will show the version as a subscript
    (except for version 1), and the relation as a prefix on
    the representation of the parent Symbol.
    """
    def __init__(self,
                 label: Union[str, 'Symbol'],
                 *,
                 namespace: str = None,
                 version: int = 1,
                 relation: str = None,
                 **kw) -> None:
        super().__init__(**kw)
        if relation is None:
            assert isinstance(label, str)
        else:
            assert isinstance(label, Symbol)
        self.label = label
        self.namespace = namespace
        self.version = version
        self.relation = relation

    def __str__(self) -> str:
        v = f'#{self.version}' if self.version > 1 else ''
        # r = f'{self.relation}:' if self.relation else ''
        r = f'{self.relation}' if self.relation else ''
        return f'{r}{self.label}{v}'

    def __eq__(self, obj) -> bool:
        """Two symbols are equal if they have the same label,
        namespace, version and relation to their label."""
        s: Symbol = obj
        return isinstance(s, Symbol) \
            and self.label == s.label \
            and self.namespace == s.namespace \
            and self.version == s.version \
            and self.relation == s.relation

    def __hash__(self) -> int:
        return hash((self.label, self.namespace,
                     self.version, self.relation))

    def __hrepr__(self, H, hrepr):
        ns = f'myia-ns-{self.namespace or "-none"}'
        rval = super().__hrepr__(H, hrepr)[ns]
        if self.relation:
            rval = rval(H.span['SymbolRelation'](self.relation))
        if isinstance(self.label, str):
            rval = rval(self.label)
        else:
            rval = rval(hrepr(self.label))
        if self.version > 1:
            rval = rval(H.span['SymbolIndex'](self.version))
        return rval


class Value(MyiaASTNode):
    """
    A literal value, like a literal integer, float or string,
    or True, False, or None. If you build an AST manually, any
    value can be put in there.

    Attributes:
        value: Some value.
    """
    def __init__(self, value, **kw):
        self.value = value
        super().__init__(**kw)

    def __str__(self) -> str:
        return repr(self.value)

    def __hrepr__(self, H, hrepr):
        return super().__hrepr__(H, hrepr)(hrepr(self.value))


class Let(MyiaASTNode):
    """
    A sequence of variable bindings followed by a body expression
    which is the Let node's return value.

    Fields:
        bindings ([(Symbol, MyiaASTNode), ...]): a list of variable
            bindings. The variable in each binding should be distinct.
        body (MyiaASTNode): The expression to return.
    """

    def __init__(self,
                 bindings: List[TupleT[Symbol, MyiaASTNode]],
                 body: MyiaASTNode,
                 **kw) -> None:
        super().__init__(**kw)
        self.bindings = bindings
        self.body = body

    def children(self) -> List[MyiaASTNode]:
        """
        Return all the variables, binding expressions, and
        then the body.
        """
        rval: List[MyiaASTNode] = []
        for a, b in self.bindings:
            rval += [a, b]
        return rval + [self.body]

    def __str__(self) -> str:
        return '(let ({}) {})'.format(
            " ".join('({} {})'.format(k, v) for k, v in self.bindings),
            self.body)

    def __hrepr__(self, H, hrepr):
        let_bindings = [
            H.div['LetBinding'](hrepr(k), hrepr(v))
            for k, v in self.bindings
        ]
        return super().__hrepr__(H, hrepr)(
            H.div['Keyword']('let'),
            H.div['LetBindings'](*let_bindings),
            H.div['Keyword']('in'),
            H.div['LetBody'](hrepr(self.body))
        )


class Lambda(MyiaASTNode):
    """
    A function definition. This is the main unit that we will
    manipulate and transform and it has a few special fields.
    Most importantly, ``gen`` is a ``GenSym`` instance that can
    be used to create fresh symbols in the context of this
    Lambda, and ``global_env`` contains the necessary bindings
    to resolve global variables in the body.

    Fields:
        args ([Symbol]): List of argument variables.
        body (MyiaASTNode): Expression that the call should return.
        gen (GenSym): Symbol factory for this Lambda.
        global_env (ParseEnv): Environment to resolve global
            variables.
        ref (Symbol): Symbol that points to this Lambda in the
            ``global_env``.
        primal (Symbol): If this Lambda is the output of ``Grad``,
            then ``primal`` points (in the ``global_env``)
            to ``Grad``'s original input Lambda. Otherwise, this
            is None.
    """
    def __init__(self,
                 args: List[Symbol],
                 body: MyiaASTNode,
                 gen: 'GenSym',
                 global_env: ParseEnv = None,
                 **kw) -> None:
        super().__init__(**kw)
        self.ref: Symbol = None
        self.args = args
        self.body = body
        self.gen = gen
        self.global_env = global_env
        self.primal: Symbol = None

    def children(self) -> List[MyiaASTNode]:
        args = cast(List[MyiaASTNode], self.args)
        return args + [self.body]

    def __str__(self) -> str:
        return '(lambda ({}) {})'.format(
            " ".join([str(arg) for arg in self.args]), str(self.body))

    def __hrepr__(self, H, hrepr):
        return super().__hrepr__(H, hrepr)(
            H.div['Keyword']('λ'),
            H.div['LambdaArguments'](*[hrepr(a) for a in self.args]),
            hrepr(self.body)
        )


class Apply(MyiaASTNode):
    """
    Function application. Note that operations like indexing or
    getting an attribute do not have their own nodes in Myia's
    AST. Instead they are applications of ``index`` or ``getattr``.

    Attributes:
        fn: Expression for the function to call.
        args: List of arguments to apply the function to.
        cannot_fail: An annotation added by the parser or
            compiler that indicates that the call is not supposed
            to fail (that is, regardless of what the user does),
            so that when it inevitably does, blame can be assigned
            properly. This is not widely used yet.
    """
    def __init__(self,
                 fn: MyiaASTNode,
                 *args: MyiaASTNode,
                 cannot_fail: bool = False,
                 **kw) -> None:
        super().__init__(**kw)
        self.fn = fn
        self.args = list(args)
        self.cannot_fail = cannot_fail

    def children(self) -> List[MyiaASTNode]:
        return [self.fn] + self.args

    def __str__(self):
        return "({} {})".format(
            str(self.fn), " ".join(str(a) for a in self.args)
        )

    def __hrepr__(self, H, hrepr):
        return super().__hrepr__(H, hrepr)(
            hrepr(self.fn),
            *[hrepr(a) for a in self.args]
        )


class Begin(MyiaASTNode):
    """
    A sequence of expressions, the last of which is the return
    value. Return values from other expressions are simply
    ignored, so if there are no side-effects this node can be
    replaced by its last element without issue.

    Attributes:
        stmts: A list of expressions, the last of which is the
            return value for Begin.
    """
    def __init__(self, stmts: List[MyiaASTNode], **kw) -> None:
        super().__init__(**kw)
        self.stmts = stmts

    def children(self) -> List[MyiaASTNode]:
        return self.stmts

    def __str__(self) -> str:
        return "(begin {})".format(" ".join(map(str, self.stmts)))

    def __hrepr__(self, H, hrepr):
        return super().__hrepr__(H, hrepr)(
            H.div['Keyword']('begin'),
            [hrepr(a) for a in self.stmts]
        )


class Tuple(MyiaASTNode):
    """
    A tuple of expressions.

    Attributes:
        values: A list of values in this tuple.
    """
    def __init__(self, values: Iterable[MyiaASTNode], **kw) -> None:
        super().__init__(**kw)
        self.values = list(values)

    def children(self) -> List[MyiaASTNode]:
        return self.values

    def __str__(self) -> str:
        return "{{{}}}".format(" ".join(map(str, self.values)))

    def __hrepr__(self, H, hrepr):
        return super().__hrepr__(H, hrepr)(
            *[hrepr(a) for a in self.values]
        )


class Closure(MyiaASTNode):
    """
    Associates a function with a list of arguments, without calling
    it. The result is a function that will concatenate the stored
    arguments to the others it will receive.

    This essentially represents a partial application.

    Attributes:
        fn: Expression for the function to call.
        args: Its first arguments.
    """
    def __init__(self,
                 fn: MyiaASTNode,
                 args: Iterable[MyiaASTNode],
                 **kw) -> None:
        super().__init__(**kw)
        self.fn = fn
        self.args = list(args)

    def children(self) -> List[MyiaASTNode]:
        return [self.fn] + self.args

    def __str__(self) -> str:
        return '(closure {} {})'.format(self.fn, " ".join(map(str, self.args)))

    def __hrepr__(self, H, hrepr):
        return super().__hrepr__(H, hrepr)(
            hrepr(self.fn),
            *[hrepr(a) for a in self.args],
            '...'
        )


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

    def sym(self, name: str) -> Symbol:
        """
        Create a unique Symbol with the given name. If one or more
        Symbols with the same name exists, the new Symbol will have
        a higher version number than any of them.
        """
        return Symbol(
            name,
            namespace=self.namespace,
            version=self.inc_version(name),
            relation=None
        )

    def rel(self, orig: Symbol, relation: str) -> Symbol:
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
        ref = f'{str(orig)}/{relation}'
        version = self.inc_version(ref)
        # print(f'{ref} {version}<br/>')
        return Symbol(
            orig,
            namespace=self.namespace,
            version=version,
            relation=relation
        )

    def __call__(self, s: Union[str, Symbol], rel: str = None) -> Symbol:
        if isinstance(rel, str):
            assert isinstance(s, Symbol)
            return self.rel(s, rel)
        else:
            return self.sym(s)


class _Assign(MyiaASTNode):
    """
    This is a "temporary" node that ``front.Parser`` uses to
    represent a variable assignment. It is then transformed
    into Let.
    """
    def __init__(self,
                 varname: Symbol,
                 value: MyiaASTNode,
                 location: Location) -> None:
        self.varname = varname
        self.value = value
        self.location = location

    def __str__(self):
        return f'(_assign {self.varname} {self.value})'


class Transformer:
    """
    Base class for Myia AST transformers.

    Upon transforming a node, ``Transformer.transform``
    transfers the original's location to the new node,
    if it doesn't have a location.

    Define methods called ``transform_<node_type>``,
    e.g. ``transform_Symbol``.
    """
    def transform(self, node, **kwargs):
        cls = node.__class__.__name__
        try:
            method = getattr(self, 'transform_' + cls)
        except AttributeError:
            raise Exception(
                "Unrecognized node type in {}: {}".format(
                    self.__class__.__name__, cls)
            )
        rval = method(node, **kwargs)
        if not rval.location:
            rval.location = node.location
        return rval
