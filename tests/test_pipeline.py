
import pytest
from myia.pipeline import merge, NS, Partial, PipelineStep, \
    PipelineDefinition, Merge, Reset, Override, DELETE, cleanup
from myia.utils import TypeMap


class OpStep(PipelineStep):

    def __init__(self, pipeline_init, op, param=0):
        super().__init__(pipeline_init)
        self.op = op
        self.param = param

    def step(self, value):
        return {'value': self.op(self.param, value)}


class OpResourceStep(PipelineStep):

    def __init__(self, pipeline_init, op):
        super().__init__(pipeline_init)
        self.op = op

    def step(self, value):
        return {'value': self.op(self.resources.param, value)}


@pytest.fixture
def op_pipeline():
    return PipelineDefinition(
        resources=dict(
            param=2
        ),
        steps=dict(
            addp=OpStep.partial(op=lambda p, x: p + x, param=1),
            mulp=OpResourceStep.partial(op=lambda p, x: p * x),
            neg=OpStep.partial(op=lambda p, x: -x),
            square=OpStep.partial(op=lambda p, x: x * x),
        )
    )


def test_merge():

    assert merge(1, 2) == 2
    assert merge([1, 2], [3, 4]) == [1, 2, 3, 4]
    assert merge((1, 2), (3, 4)) == (1, 2, 3, 4)
    assert merge({1, 2}, {3, 4}) == {1, 2, 3, 4}

    a = dict(a=1, b=2, c=dict(d=3, e=[4, 5]))
    b = dict(a=1, b=2, c=dict(e=[6, 7], f=8))
    c = dict(a=1, b=2, c=dict(d=3, e=[4, 5, 6, 7], f=8))

    assert merge(a, b) == c

    dlt = dict(c=DELETE, d=3)
    assert merge(a, dlt) == dict(a=1, b=2, d=3)


def test_merge_subclass():

    tm = TypeMap({int: "int"})
    mtm = merge(tm, {str: "str"})
    assert isinstance(mtm, TypeMap)
    assert mtm == TypeMap({int: "int", str: "str"})


def test_merge_modes():

    for x, y in [({1, 2}, {3, 4}),
                 ([1, 2], [3, 4]),
                 ((1, 2), (3, 4))]:

        assert merge(x, y, mode='reset') == y
        assert merge(x, Reset(y)) == y
        assert merge(x, Reset(y), mode='merge') == y

        assert merge(x, y, mode='override') == y
        assert merge(x, Override(y)) == y
        assert merge(x, Override(y), mode='merge') == y

    a = {'a': 1}
    b = {'b': 2}
    c = {'a': 1, 'b': 2}

    assert merge(a, b, mode='reset') == b
    assert merge(a, b, mode='override') == c

    a = {'a': [1, 2], 'b': [3, 4]}
    b = {'a': [5, 6], 'b': Override([7, 8])}
    c = {'a': [1, 2, 5, 6], 'b': [7, 8]}
    d = {'a': [5, 6], 'b': [7, 8]}

    assert merge(a, b) == c
    assert merge(a, b, mode='override') == d


def test_cleanup():
    a = dict(a=1, b=[2, Merge(3)], c=Override(4), d=DELETE)
    assert cleanup(a) == dict(a=1, b=[2, 3], c=4)


def test_cleanup_subclass():
    a = TypeMap({int: Merge("int")})
    ca = cleanup(a)
    assert isinstance(ca, TypeMap)
    assert ca == TypeMap({int: "int"})


def test_NS():
    ns = NS(x=1, y=2)

    assert ns.x == 1
    assert ns.y == 2

    ns.a = 3
    assert ns.a == 3
    assert ns['a'] == 3

    ns['b'] = 4
    assert ns['b'] == 4
    assert ns.b == 4

    assert repr(ns) == 'NS(x=1, y=2, a=3, b=4)'


def test_Partial():

    def f(x, y):
        return x + y

    p1 = Partial(f, x=10)
    p2 = Partial(f, y=20)

    assert p1(y=3) == 13
    assert p2(x=3) == 23
    assert merge(p1, p2)() == 30
    assert merge(p1, {'y': 20})() == 30

    with pytest.raises(TypeError):
        Partial(f, z=10)

    p3 = Reset(Partial(f, y=20))
    with pytest.raises(TypeError):
        merge(p1, p3)()

    p4 = Partial(f, x=[1, 2])
    p5 = Partial(f, x=[3, 4])

    assert merge(p4, p5)(y=[10]) == [1, 2, 3, 4, 10]
    assert merge(p5, p4)(y=[10]) == [3, 4, 1, 2, 10]

    p6 = Override(Partial(f, x=[3, 4]))
    assert merge(p4, p6)(y=[10]) == [3, 4, 10]

    def g(x, y):
        return x * y

    p7 = Partial(g, y=20)
    with pytest.raises(ValueError):
        merge(p1, p7)

    p8 = Override(Partial(g, y=20))
    assert merge(p1, p8)() == 200

    def h(**kw):
        return kw

    p9 = Partial(h, x=10)
    assert p9(y=20, z=30) == dict(x=10, y=20, z=30)

    str(p9), repr(p9)


def test_Partial_class():

    class C:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __call__(self):
            return self.x + self.y

    p1 = Partial(C, x=10)
    p2 = Partial(C, y=20)

    assert p1(y=3)() == 13
    assert p2(x=3)() == 23
    assert merge(p1, p2)()() == 30

    with pytest.raises(TypeError):
        Partial(C, z=10)


def test_Pipeline(op_pipeline):
    pdef = op_pipeline

    pip = pdef.make()
    assert pip(value=3) == {'value': 64}

    assert pip.steps.addp(value=3) == {'value': 4}
    assert pip.steps.mulp(value=3) == {'value': 8}
    assert pip.steps.neg(value=3) == {'value': -8}
    assert pip.steps.square(value=3) == {'value': 64}

    assert pip['addp'](value=3) == {'value': 4}
    assert pip['mulp'](value=3) == {'value': 8}
    assert pip['neg'](value=3) == {'value': -8}
    assert pip['square'](value=3) == {'value': 64}

    assert pip['mulp':](value=3) == {'value': 36}
    assert pip[:'mulp'](value=3) == {'value': 8}
    assert pip[:-1](value=3) == {'value': -8}
    assert pip['mulp':'neg'](value=3) == {'value': -6}

    assert pip['!square'](value=3) == {'value': -8}
    assert pip['mulp':'!neg'](value=3) == {'value': 6}

    assert pdef['mulp':'neg'].make()(value=3) == {'value': -6}


def test_Pipeline_configure(op_pipeline):
    pdef = op_pipeline

    pip = pdef.configure(addp=Merge(param=2)).make()
    assert pip(value=3) == {'value': 100}

    pip = pdef.configure({'addp.param': 2}).make()
    assert pip(value=3) == {'value': 100}

    pip = pdef.configure(
        {'addp.param': 2},
        addp=Merge(op=lambda p, x: p - x)
    ).make()
    assert pip(value=3) == {'value': 4}

    pip = pdef.configure(addp=Reset(op=lambda p, x: p - x, param=2)).make()
    assert pip(value=3) == {'value': 4}

    with pytest.raises(TypeError):
        pdef.configure(addp=Reset(param=2)).make()

    pip = pdef.configure(mulp=False).make()
    assert pip(value=3) == {'value': 16}

    pip = pdef.configure(mulp=False).configure(mulp=True).make()
    assert pip(value=3) == {'value': 64}

    pip = pdef.configure(mulp=False).configure(addp=False).make()
    assert pip(value=3) == {'value': 9}

    pip = pdef.configure(param=3).make()
    assert pip(value=3) == {'value': 144}

    with pytest.raises(KeyError):
        pdef.configure(quack=[1, 2])

    pdef2 = pdef.configure_resources(quack=[1, 2])
    assert pdef2.make().resources.quack == [1, 2]
    assert pdef2.configure(quack=Merge([3])).make().resources.quack \
        == [1, 2, 3]
    assert pdef2.configure(quack=[3]).make().resources.quack == [3]


def test_Pipeline_insert(op_pipeline):
    pdef = op_pipeline

    half = OpStep.partial(op=lambda p, x: x / p, param=2)

    pip = pdef.insert_before(half=half).make()
    assert pip(value=3) == {'value': 25}

    pip = pdef.insert_after(half=half).make()
    assert pip(value=3) == {'value': 32}

    pip = pdef.insert_before('addp', half=half).make()
    assert pip(value=3) == {'value': 25}

    pip = pdef.insert_after('addp', half=half).make()
    assert pip(value=3) == {'value': 16}

    pip = pdef.insert_before('square', half=half).make()
    assert pip(value=3) == {'value': 16}

    pip = pdef.insert_after('square', half=half).make()
    assert pip(value=3) == {'value': 32}


def test_Pipeline_select(op_pipeline):
    pdef = op_pipeline

    pip = pdef.select('addp', 'neg').make()
    assert pip(value=3) == {'value': -4}

    pip = pdef.select('mulp', 'square').make()
    assert pip(value=3) == {'value': 36}

    pip = pdef.select('square', 'mulp').make()
    assert pip(value=3) == {'value': 18}
