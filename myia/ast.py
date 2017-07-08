

from uuid import uuid1
from copy import copy
import textwrap


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
    def at(self, location):
        rval = copy(self)
        rval.location = location
        return rval


class Symbol(MyiaASTNode):
    def __init__(self, label, *, location=None):
        assert isinstance(label, str)
        self.label = label
        self.location = location

    def __str__(self):
        return self.label


class Literal(MyiaASTNode):
    def __init__(self, value, *, location=None):
        self.value = value
        self.location = location

    def __str__(self):
        return repr(self.value)


class LetRec(MyiaASTNode):
    def __init__(self, bindings, body, *, location=None):
        self.bindings = bindings
        self.location = location
        self.body = body

    def __str__(self):
        return '(letrec ({}) {})'.format(
            " ".join('({} {})'.format(k, v) for k, v in self.bindings),
            self.body)


class Lambda(MyiaASTNode):
    def __init__(self, label, args, body, *, location=None):
        self.label = label
        self.args = args
        self.body = body
        self.location = location

    def __str__(self):
        return '(lambda ({}) {})'.format(
            " ".join([str(arg) for arg in self.args]), str(self.body))


class If(MyiaASTNode):
    def __init__(self, cond, t, f, *, location=None):
        self.cond = cond
        self.t = t
        self.f = f
        self.location = location

    def __str__(self):
        return '(if {} {} {})'.format(self.cond, self.t, self.f)


class Apply(MyiaASTNode):
    def __init__(self, fn, *args, location=None):
        self.fn = fn
        self.args = args
        self.location = location

    def __str__(self):
        return "({} {})".format(str(self.fn), " ".join(str(a) for a in self.args))


class Begin(MyiaASTNode):
    def __init__(self, stmts, location=None):
        self.stmts = stmts
        self.location = location

    def __str__(self):
        return "(begin {})".format(" ".join(map(str, self.stmts)))
