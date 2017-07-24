from uuid import uuid4 as uuid
from copy import copy
import textwrap
import traceback
import os.path


_css_path = f'{os.path.dirname(__file__)}/ast.css'
_css = None
__save_trace__ = False


class Location:
    def __init__(self, url, line, column):
        self.url = url
        self.line = line
        self.column = column

    def __str__(self):
        return '{}@{}:{}'.format(self.url, self.line, self.column)

    def traceback(self):
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


def _get_location(x):
    if isinstance(x, MyiaASTNode):
        return x.location
    elif isinstance(x, Location) or x is None:
        return x
    else:
        raise TypeError(f'{x} is not a location')


class MyiaASTNode:
    @classmethod
    def __hrepr_resources__(cls, H):
        global _css
        if _css is None:
            _css = open(_css_path).read()
        return H.style(_css)

    def __init__(self, location=None):
        self.location = _get_location(location)
        if __save_trace__:
            # TODO: make sure that we're always removing the right
            # number of entries
            self.trace = traceback.extract_stack()[:-2]
        else:
            self.trace = None

    def at(self, location):
        rval = copy(self)
        rval.location = _get_location(location)
        return rval

    def children(self):
        return []

    def __repr__(self):
        return str(self)


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
    def __init__(self, label, *,
                 namespace=None,
                 version=1,
                 relation=None,
                 **kw):
        super().__init__(**kw)
        if relation is None:
            assert isinstance(label, str)
        else:
            assert isinstance(label, Symbol)
        self.label = label
        self.namespace = namespace
        self.version = version
        self.relation = relation

    def __str__(self):
        v = f'#{self.version}' if self.version > 1 else ''
        r = f'{self.relation}:' if self.relation else ''
        return f'{r}{self.label}{v}'

    def __eq__(self, s):
        return isinstance(s, Symbol) \
            and self.label == s.label \
            and self.namespace == s.namespace \
            and self.version == s.version \
            and self.relation == s.relation

    def __hash__(self):
        return hash((self.label, self.namespace,
                     self.version, self.relation))

    def __hrepr__(self, H, hrepr):
        ns = f'myia-ns-{self.namespace or "-none"}'
        rval = H.div['Symbol', ns]
        if self.relation:
            rval = rval(H.span['SymbolRelation'])
        if isinstance(self.label, str):
            rval = rval(self.label)
        else:
            rval = rval(hrepr(self.label))
        if self.version <= 1:
            return rval
        else:
            return rval(H.span['SymbolIndex'](self.version))


class Value(MyiaASTNode):
    def __init__(self, value, **kw):
        self.value = value
        super().__init__(**kw)

    def __str__(self):
        return repr(self.value)

    def __hrepr__(self, H, hrepr):
        return H.div['Value'](hrepr(self.value))


class Let(MyiaASTNode):
    def __init__(self, bindings, body, **kw):
        super().__init__(**kw)
        self.bindings = bindings
        self.body = body

    def children(self):
        rval = []
        for a, b in self.bindings:
            rval += [a, b]
        return rval + [self.body]

    def __str__(self):
        return '(let ({}) {})'.format(
            " ".join('({} {})'.format(k, v) for k, v in self.bindings),
            self.body)

    def __hrepr__(self, H, hrepr):
        let_bindings = [
            H.div['LetBinding'](hrepr(k), hrepr(v))
            for k, v in self.bindings
        ]
        return H.div['Let'](
            H.div['Keyword']('let'),
            H.div['LetBindings'](*let_bindings),
            H.div['Keyword']('in'),
            H.div['LetBody'](hrepr(self.body))
        )


class Lambda(MyiaASTNode):
    def __init__(self, label, args, body, **kw):
        super().__init__(**kw)
        self.label = label
        self.args = args
        self.body = body

    def children(self):
        return self.args + [self.body]

    def __str__(self):
        return '(lambda ({}) {})'.format(
            " ".join([str(arg) for arg in self.args]), str(self.body))

    def __hrepr__(self, H, hrepr):
        return H.div['Lambda'](
            H.div['Keyword']('λ'),
            H.div['LambdaArguments'](*[hrepr(a) for a in self.args]),
            hrepr(self.body)
        )


class If(MyiaASTNode):
    def __init__(self, cond, t, f, **kw):
        super().__init__(**kw)
        self.cond = cond
        self.t = t
        self.f = f

    def children(self):
        return [self.cond, self.t, self.f]

    def __str__(self):
        return '(if {} {} {})'.format(self.cond, self.t, self.f)

    def __hrepr__(self, H, hrepr):
        return H.div['If'](
            H.div['IfCond'](hrepr(self.cond)),
            H.div['IfThen'](hrepr(self.t)),
            H.div['IfElse'](hrepr(self.f))
        )


class Apply(MyiaASTNode):
    def __init__(self, fn, *args, cannot_fail=False, **kw):
        super().__init__(**kw)
        self.fn = fn
        self.args = tuple(args)
        # Boilerplate calls added by the parser should be
        # annotated cannot_fail if they are not supposed to fail,
        # so that when they inevitably do, blame can be assigned properly.
        self.cannot_fail = cannot_fail

    def children(self):
        return (self.fn,) + self.args

    def __str__(self):
        return "({} {})".format(
            str(self.fn), " ".join(str(a) for a in self.args)
        )

    def __hrepr__(self, H, hrepr):
        return H.div['Apply'](hrepr(self.fn), *[hrepr(a) for a in self.args])


class Begin(MyiaASTNode):
    def __init__(self, stmts, **kw):
        super().__init__(**kw)
        self.stmts = stmts

    def children(self):
        return self.stmts

    def __str__(self):
        return "(begin {})".format(" ".join(map(str, self.stmts)))

    def __hrepr__(self, H, hrepr):
        return H.div['Begin'](
            H.div['Keyword']('begin'),
            [hrepr(a) for a in self.stmts]
        )


class Tuple(MyiaASTNode):
    def __init__(self, values, **kw):
        super().__init__(**kw)
        self.values = list(values)

    def children(self):
        return self.values

    def __str__(self):
        return "{{{}}}".format(" ".join(map(str, self.values)))

    def __hrepr__(self, H, hrepr):
        return H.div['Tuple'](*[hrepr(a) for a in self.values])


class Closure(MyiaASTNode):
    def __init__(self, fn, args, **kw):
        super().__init__(**kw)
        self.fn = fn
        self.args = list(args)

    def children(self):
        return [self.fn] + self.args

    def __str__(self):
        return '(closure {} {})'.format(self.fn, " ".join(map(str, self.args)))

    def __hrepr__(self, H, hrepr):
        return H.div['Closure'](hrepr(self.fn),
                                *[hrepr(a) for a in self.args], '...')


class GenSym:
    def __init__(self, namespace=None):
        self.varcounts = {}
        self.namespace = namespace or uuid()

    def inc_version(self, ref):
        if ref in self.varcounts:
            self.varcounts[ref] += 1
            return self.varcounts[ref]
        else:
            self.varcounts[ref] = 1
            return 1

    def sym(self, name, reference_name=None):
        return Symbol(
            name,
            namespace=self.namespace,
            version=self.inc_version(reference_name or name),
            relation=None
        )

    def rel(self, orig, relation):
        ref = f'{str(orig)}/{relation}'
        version = self.inc_version(ref)
        return Symbol(
            orig,
            namespace=self.namespace,
            version=version,
            relation=relation
        )


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
