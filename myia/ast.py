

from uuid import uuid1
from copy import copy
import textwrap
import traceback


__save_trace__ = False # __debug__


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
                code, caret = textwrap.dedent(raw_code + '\n' + raw_caret).split("\n")
            return '  File "{}", line {}, column {}\n    {}\n    {}'.format(
                self.url, self.line, self.column, code, caret)
        except FileNotFoundError as e:
            return '  File "{}", line {}, column {}'.format(
                self.url, self.line, self.column)


class MyiaASTNode:
    def __init__(self, location=None):
        self.location = location
        if __save_trace__:
            # TODO: make sure that we're always removing the right
            # number of entries
            self.trace = traceback.extract_stack()[:-2]
        else:
            self.trace = None

    def at(self, location):
        rval = copy(self)
        rval.location = location
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
        return self.label # + ':' + (self.namespace or '???')

    def __descr__(self, _):
        *first, last = self.label.split("#")
        if len(first) == 0 or first == [""]:
            return ({"Symbol"}, self.label, ({"Namespace"}, self.namespace or "???"))
        else:
            return ({"Symbol"}, "#".join(first), ({"SymbolIndex"}, last),
                    ({"Namespace"}, self.namespace or "???"))


class Literal(MyiaASTNode):
    def __init__(self, value, **kw):
        self.value = value
        super().__init__(**kw)

    def __str__(self):
        return repr(self.value)

    def __descr__(self, recurse):
        return ({"Literal"}, recurse(self.value))


class LetRec(MyiaASTNode):
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
        return '(letrec ({}) {})'.format(
            " ".join('({} {})'.format(k, v) for k, v in self.bindings),
            self.body)

    def __descr__(self, recurse):
        return ({"LetRec"},
                ({"Keyword"}, "letrec"),
                ({"LetRecBindings"},
                 *[[{"LetRecBinding"}, recurse(k), recurse(v)] for k, v in self.bindings]),
                ({"Keyword"}, "in"),
                ({"LetRecBody"}, recurse(self.body)))



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

    def __descr__(self, recurse):
        return ({"Lambda"},
                ({"Keyword"}, "λ"),
                ({"LambdaArguments"}, *[recurse(a) for a in self.args]),
                recurse(self.body))


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

    def __descr__(self, recurse):
        return ({"If"},
                ({"IfCond"}, recurse(self.cond)),
                ({"IfTrue"}, recurse(self.t)),
                ({"IfFalse"}, recurse(self.f)))


class Apply(MyiaASTNode):
    def __init__(self, fn, *args, cannot_fail=False, **kw):
        super().__init__(**kw)
        self.fn = fn
        self.args = tuple(args)
        # Boilerplate calls added by the parser should be annotated cannot_fail
        # if they are not supposed to fail, so that when they inevitably do, blame
        # can be assigned properly.
        self.cannot_fail = cannot_fail

    def children(self):
        return (self.fn,) + self.args

    def __str__(self):
        return "({} {})".format(str(self.fn), " ".join(str(a) for a in self.args))

    def __descr__(self, recurse):
        return ({"Apply"},
                recurse(self.fn),
                *[recurse(a) for a in self.args])


class Begin(MyiaASTNode):
    def __init__(self, stmts, **kw):
        super().__init__(**kw)
        self.stmts = stmts

    def children(self):
        return self.stmts

    def __str__(self):
        return "(begin {})".format(" ".join(map(str, self.stmts)))

    def __descr__(self, recurse):
        return ({"Begin"},
                ({"Keyword"}, "begin"),
                [recurse(a) for a in self.stmts])

class Tuple(MyiaASTNode):
    def __init__(self, values, **kw):
        super().__init__(**kw)
        self.values = list(values)

    def children(self):
        return self.values

    def __str__(self):
        return "{{{}}}".format(" ".join(map(str, self.values)))

    def __descr__(self, recurse):
        return ({"Tuple"}, *[recurse(a) for a in self.values])

class Closure(MyiaASTNode):
    def __init__(self, fn, args, **kw):
        super().__init__(**kw)
        self.fn = fn
        self.args = list(args)

    def children(self):
        return [self.fn] + self.args

    def __str__(self):
        return '(closure {} {})'.format(self.fn, " ".join(map(str, self.args)))

    def __descr__(self, recurse):
        return ({"Closure"},
                recurse(self.fn),
                *[recurse(a) for a in self.args])
