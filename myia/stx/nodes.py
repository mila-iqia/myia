"""
Myia's AST.

* Definitions for all node types.
* Symbol handling: GenSym and ParseEnv.
* Base class for Transformers.
"""

from typing import \
    List, Tuple as TupleT, Iterable, Dict, Set, Union, \
    cast, TypeVar, Any

from copy import copy
import traceback
from ..util import HReprBase
from .about import Location, top as about_top


__save_trace__ = False


Locatable = Union['MyiaASTNode', 'Location', None]
LHS = Union['Symbol', 'Tuple']
Binding = TupleT[LHS, 'MyiaASTNode']
Bindings = List[Binding]


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
    def __init__(self, **kw) -> None:
        if __save_trace__:
            frames = traceback.extract_stack()
            frame = frames.pop()
            # We skip all frames from helper functions defined
            # in this file.
            while frames and frame.filename == __file__:  # type: ignore
                frame = frames.pop()
            self.trace = Location(
                frame.filename,  # type: ignore
                frame.lineno,  # type: ignore
                -1,
                None
            )
        else:
            self.trace = None
        self.about = about_top()
        self.annotations: Set[str] = set()

    def children(self) -> List['MyiaASTNode']:
        return []

    def find_location(self) -> Location:
        node = self
        while getattr(node, 'about', None):
            node = node.about.node
        if isinstance(node, Location):
            return node
        return None

    def __repr__(self) -> str:
        return str(self)

    def __hrepr__(self, H, hrepr):
        rval = H.replObject[self.__class__.__name__][f'pyid-{id(self)}']
        rval = rval({"obj-id": id(self)})
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

    def copy(self, preserve_about=True) -> 'Symbol':
        rval = Symbol(self.label,
                      namespace=self.namespace,
                      version=self.version,
                      relation=self.relation)
        if preserve_about:
            rval.about = self.about
        return rval

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

    def __eq__(self, other):
        return isinstance(other, Value) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

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
                 bindings: Bindings,
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
                 global_env: 'ParseEnv' = None,
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


class _Assign(MyiaASTNode):
    """
    This is a "temporary" node that ``front.Parser`` uses to
    represent a variable assignment. It is then transformed
    into Let.
    """
    def __init__(self,
                 varname: LHS,
                 value: MyiaASTNode) -> None:
        self.varname = varname
        self.value = value

    def __str__(self):
        return f'(_assign {self.varname} {self.value})'


# For type annotations:
from .env import GenSym, ParseEnv
