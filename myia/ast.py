from uuid import uuid1
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
    def __init__(self, label, *, namespace=None, **kw):
        super().__init__(**kw)
        assert isinstance(label, str)
        self.label = label
        self.namespace = namespace

    def __str__(self):
        return self.label

    def __eq__(self, s):
        return isinstance(s, Symbol) \
            and self.label == s.label \
            and self.namespace == s.namespace

    def __hash__(self):
        return hash((self.label, self.namespace))

    def __hrepr__(self, H, hrepr):
        *first, last = self.label.split("#")
        ns = f'myia-ns-{self.namespace or "-none"}'
        if len(first) == 0 or first == [""]:
            return H.div['Symbol', ns](
                self.label,
            )
        else:
            return H.div['Symbol', ns](
                '#'.join(first),
                H.span['SymbolIndex'](last),
            )


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
