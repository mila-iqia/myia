from typing import \
    List, Tuple as TupleT, Iterable, Dict, Set, Union, \
    cast, TypeVar

from uuid import uuid4 as uuid
from copy import copy
import textwrap
import traceback
from .buche import HReprBase


__save_trace__ = False


Locatable = Union['MyiaASTNode', 'Location', None]


class Location:
    def __init__(self, url: str, line: int, column: int) -> None:
        self.url = url
        self.line = line
        self.column = column

    def __str__(self) -> str:
        return '{}@{}:{}'.format(self.url, self.line, self.column)

    def traceback(self) -> str:
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
    if isinstance(x, MyiaASTNode):
        return x.location
    elif isinstance(x, Location) or x is None:
        return x
    else:
        raise TypeError(f'{x} is not a location')


T = TypeVar('T', bound='MyiaASTNode')


class MyiaASTNode(HReprBase):
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
        rval = copy(self)
        rval.location = _get_location(location)
        return rval

    def children(self) -> List['MyiaASTNode']:
        return []

    def __repr__(self) -> str:
        return str(self)

    def __hrepr__(self, H, hrepr):
        rval = H.div[self.__class__.__name__]
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
            this must be a symbol.
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
    def __init__(self, value, **kw):
        self.value = value
        super().__init__(**kw)

    def __str__(self) -> str:
        return repr(self.value)

    def __hrepr__(self, H, hrepr):
        return super().__hrepr__(H, hrepr)(hrepr(self.value))


class Let(MyiaASTNode):
    def __init__(self,
                 bindings: List[TupleT[Symbol, MyiaASTNode]],
                 body: MyiaASTNode,
                 **kw) -> None:
        super().__init__(**kw)
        self.bindings = bindings
        self.body = body

    def children(self) -> List[MyiaASTNode]:
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
    def __init__(self,
                 args: List[Symbol],
                 body: MyiaASTNode,
                 gen: 'GenSym',
                 **kw) -> None:
        super().__init__(**kw)
        self.ref: Symbol = None
        self.args = args
        self.body = body
        self.gen = gen

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


class If(MyiaASTNode):
    def __init__(self,
                 cond: MyiaASTNode,
                 t: MyiaASTNode,
                 f: MyiaASTNode,
                 **kw) -> None:
        super().__init__(**kw)
        self.cond = cond
        self.t = t
        self.f = f

    def children(self) -> List[MyiaASTNode]:
        return [self.cond, self.t, self.f]

    def __str__(self) -> str:
        return '(if {} {} {})'.format(self.cond, self.t, self.f)

    def __hrepr__(self, H, hrepr):
        return super().__hrepr__(H, hrepr)(
            H.div['IfCond'](hrepr(self.cond)),
            H.div['IfThen'](hrepr(self.t)),
            H.div['IfElse'](hrepr(self.f))
        )


class Apply(MyiaASTNode):
    def __init__(self,
                 fn: MyiaASTNode,
                 *args: MyiaASTNode,
                 cannot_fail: bool = False,
                 **kw) -> None:
        super().__init__(**kw)
        self.fn = fn
        self.args = list(args)
        # Boilerplate calls added by the parser should be
        # annotated cannot_fail if they are not supposed to fail,
        # so that when they inevitably do, blame can be assigned properly.
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
        if isinstance(s, str):
            return self.sym(s)
        else:
            return self.rel(s, rel)


class _Assign(MyiaASTNode):
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
