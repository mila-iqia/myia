import types
from types import GeneratorType

import pytest
from ovld import ovld

from myia.abstract import data, utils as autils
from myia.abstract.data import ANYTHING, Placeholder
from myia.abstract.map import MapError
from myia.infer.algo import Merge, Require, RequireAll, Unify, infer

from ..common import A, Un


class HasAbstract:
    def __init__(self, abstract=None):
        self.abstract = abstract


class Parameter(HasAbstract):
    def __init__(self, idx, abstract=None):
        self.idx = idx
        super().__init__(abstract)


class Leaf(HasAbstract):
    def __init__(self, name, type):
        self.name = name
        self.type = type
        super().__init__()

    def infer(self, unif):
        return self.type


class Binary(HasAbstract):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        super().__init__()


class Ternary(HasAbstract):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        super().__init__()


class First(Binary):
    def infer(self, unif):
        return (yield Require(self.a))


class Second(Binary):
    def infer(self, unif):
        return (yield Require(self.b))


class Combine(Binary):
    def infer(self, unif):
        return (yield Merge(self.a, self.b))


class CombineU(Binary):
    def infer(self, unif):
        return (yield Unify(self.a, self.b))


class Tuple(HasAbstract):
    def __init__(self, *elements):
        self.elements = elements
        super().__init__()

    def infer(self, unif):
        elems = yield RequireAll(*self.elements)
        return A(tuple(elems))


class Switch(Ternary):
    def infer(self, unif):
        cond = yield Require(self.a)
        if cond.tracks.value is True:
            return (yield Require(self.b))
        elif cond.tracks.value is False:
            return (yield Require(self.c))
        else:
            return (yield Merge(self.b, self.c))


# class Switch2(Ternary):
#     def infer(self):
#         cond = yield Require(self.a)
#         if cond.tracks.value is True:
#             return (yield Require(self.b))
#         elif cond.tracks.value is False:
#             return (yield Require(self.c))
#         else:
#             return (yield Unify(self.b, self.c))


class CallUnify(HasAbstract):
    def __init__(self, op, *args):
        self.op = op
        self.args = list(args)
        super().__init__()

    def infer(self, unif):
        op = self.op
        if len(self.args) != len(op.args):
            raise TypeError("Wrong number of arguments")
        args = yield RequireAll(*self.args)
        for argt, arg in zip(op.args, args):
            autils.unify(argt, arg, U=unif)
        return autils.reify(self.op.out, unif=unif.canon)


class Expression(HasAbstract):
    def __init__(self, expr):
        self.expr = expr
        self.cache = {}
        super().__init__()

    def specialize(self, args):
        if args not in self.cache:
            self.cache[args] = cp(self.expr, args)
        return self.cache[args]


@ovld
def cp(self, x: Parameter, args):
    return Parameter(idx=x.idx, abstract=args[x.idx])
    # return Leaf(x.idx, args[x.idx])


@ovld
def cp(self, x: Binary, args):  # noqa: F811
    return type(x)(a=self(x.a, args), b=self(x.b, args))


@ovld
def cp(self, xs: Tuple, args):  # noqa: F811
    return Tuple(*[self(x, args) for x in xs])


@ovld
def cp(self, call: CallUnify, args):  # noqa: F811
    return CallUnify(call.op, *[self(x, args) for x in call.args])


class CallExpr(HasAbstract):
    def __init__(self, expr, *args):
        self.expr = expr
        self.args = args
        super().__init__()

    def infer(self, unif):
        args = yield RequireAll(*self.args)
        expr = self.expr.specialize(args)
        return (yield Require(expr))


def eng(node, unif):
    assert node is not None
    assert not isinstance(node, (data.AbstractValue, data.GenericBase))
    x = node.infer(unif)
    if isinstance(x, GeneratorType):
        return (yield from x)
    else:
        return x


def L(name, *values):
    return Leaf(name, A(*values))


def U(name, *values):
    return Leaf(name, Un(*values))


def test_algo_1():
    node = First(Second(L("a", 1), L("b", 2)), L("c", 3))
    assert infer(engine=eng, node=node) is A(2)


def test_switch():
    node = Switch(L("a", True), L("b", 5), L("c", 6))
    assert infer(engine=eng, node=node) is A(5)

    node = Switch(L("a", False), L("b", 5), L("c", 6))
    assert infer(engine=eng, node=node) is A(6)

    node = Switch(L("a", ANYTHING), L("b", 5), L("c", 5))
    assert infer(engine=eng, node=node) is A(5)


def test_tuple():
    node = Tuple(L("a", 1), L("a", 2), L("a", 3))
    assert infer(engine=eng, node=node) is A(1, 2, 3)


def test_merge():
    node = Combine(L("a", 5), L("b", 5))
    assert infer(engine=eng, node=node) is A(5)

    with pytest.raises(MapError):
        node = Combine(L("a", 5), L("b", 6))
        infer(engine=eng, node=node)

    node = Combine(U("a", 5), U("b", 6))
    assert infer(engine=eng, node=node) is Un(5, 6)


def test_unify():
    node = CombineU(L("a", 5), L("b", 5))
    assert infer(engine=eng, node=node) is A(5)

    with pytest.raises(MapError):
        node = CombineU(L("a", 5), L("b", 6))
        infer(engine=eng, node=node)

    with pytest.raises(MapError):
        node = CombineU(U("a", 5), U("b", 6))
        infer(engine=eng, node=node)

    node = CombineU(U("a", 5), U("b", 5))
    assert infer(engine=eng, node=node) is Un(5)


def test_unify_samenode():
    a = L("a", 5)
    node = CombineU(a, a)
    assert infer(engine=eng, node=node) is A(5)


def test_generic():
    ph = Placeholder()
    node = Tuple(
        Combine(L("a", 13), L("b", ph)),
        Combine(L("c", ph), L("d", 31)),
    )
    with pytest.raises(MapError):
        print(infer(engine=eng, node=node))


def test_call_unify():
    var = data.Placeholder()
    node = CallUnify(
        data.AbstractFunction(
            (var, var, var), tracks={"interface": types.FunctionType}
        ),
        L("a", int),
        L("b", int),
    )
    assert infer(engine=eng, node=node) is A(int)


def test_call_unify_bad():
    var = data.Placeholder()
    node = CallUnify(
        data.AbstractFunction(
            (var, var, var), tracks={"interface": types.FunctionType}
        ),
        L("a", int),
        L("b", float),
    )
    with pytest.raises(MapError):
        assert infer(engine=eng, node=node) is A(int)


def test_call_unify2():
    var = data.Generic("x")
    node = CallUnify(
        data.AbstractFunction(
            (var, A(int), var), tracks={"interface": types.FunctionType}
        ),
        L("a", int),
        L("b", int),
    )
    assert infer(engine=eng, node=node) is A(int)

    node = CallUnify(
        data.AbstractFunction(
            (var, A(int), var), tracks={"interface": types.FunctionType}
        ),
        L("a", float),
        L("b", int),
    )
    assert infer(engine=eng, node=node) is A(float)


def test_call_unify3():
    var = data.Generic("x")
    ftyp1 = data.AbstractFunction(
        (var, var, var), tracks={"interface": types.FunctionType}
    )
    ftyp2 = data.AbstractFunction(
        (A(int), A(float), A(float)), tracks={"interface": types.FunctionType}
    )
    node = CallUnify(
        ftyp2,
        CallUnify(ftyp1, L("a", int), L("b", int)),
        CallUnify(ftyp1, L("c", float), L("d", float)),
    )
    assert infer(engine=eng, node=node) is A(float)

    with pytest.raises(MapError):
        node = CallUnify(
            ftyp2,
            CallUnify(ftyp1, L("a", int), L("b", float)),
            CallUnify(ftyp1, L("c", float), L("d", float)),
        )
        assert infer(engine=eng, node=node) is A(float)

    with pytest.raises(MapError):
        node = CallUnify(
            ftyp2,
            CallUnify(ftyp1, L("c", float), L("d", float)),
            CallUnify(ftyp1, L("a", int), L("b", int)),
        )
        assert infer(engine=eng, node=node) is A(float)


def test_call_expr():
    var = data.Generic("x")
    ftyp1 = data.AbstractFunction(
        (var, var, var), tracks={"interface": types.FunctionType}
    )
    ftyp2 = data.AbstractFunction(
        (A(int), A(float), A(float)), tracks={"interface": types.FunctionType}
    )

    p0 = Parameter(0)
    p1 = Parameter(1)
    expr = Expression(CallUnify(ftyp1, p0, p1))

    node = CallUnify(
        ftyp2,
        CallExpr(expr, L("a", int), L("b", int)),
        CallExpr(expr, L("c", float), L("d", float)),
    )

    assert infer(engine=eng, node=node) is A(float)


def test_recurse():
    var = data.Generic("x")
    ftyp = data.AbstractFunction(
        (var, var, var), tracks={"interface": types.FunctionType}
    )
    node = CallUnify(ftyp, L("a", int), None)
    node.args[1] = node

    assert infer(engine=eng, node=node) is A(int)


def test_recurse_2():
    var1 = data.Generic("x")
    var2 = data.Generic("y")
    ftyp = data.AbstractFunction(
        (var1, var2, var1, var2), tracks={"interface": types.FunctionType}
    )
    # node = CallUnify(ftyp, L("a", int), L("b", float), L("c", float))
    node = CallUnify(ftyp, L("a", int), L("b", float), None)
    node.args[2] = node

    with pytest.raises(MapError):
        print(infer(engine=eng, node=node))
