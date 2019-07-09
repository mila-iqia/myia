from pytest import mark

from myia.pipeline import standard_debug_pipeline
from myia.ir import manage


cconv_pipeline = standard_debug_pipeline \
    .select('parse', 'resolve', 'cconv', 'export')


def check_no_free_variables(root):
    mng = manage(root)
    for g, nodes in mng.nodes.items():
        if not g:
            continue
        if g.parent is not None:
            raise Exception(f'Nested graph detected: {g}')
        for node in nodes:
            assert node.graph is g
            for inp in node.inputs:
                if inp.graph is not None and inp.graph is not g:
                    raise Exception(f'Free variable detected: {node}')


def cconv(*arglists):

    def decorate(fn):
        def run_test(args):
            result_py = fn(*args)
            res = cconv_pipeline.make()(input=fn)
            check_no_free_variables(res['graph'])
            result_final = res['output'](*args)
            assert result_py == result_final

        m = mark.parametrize('args', arglists)(run_test)
        m.__orig__ = fn
        return m

    return decorate


@cconv((12, 34))
def test_straight(x, y):
    return x * x + y * y


@cconv((12,))
def test_simple_closure(x):
    def g():
        return x

    return g()


@cconv((7, 3), (1, 3))
def test_max(x, y):
    if x > y:
        return x
    else:
        return y


@cconv((53,))
def test_deep_nesting(x):
    def f(y):
        def g(z):
            def h():
                return y + z
            return h
        return g(x)
    a = f(x + 1)
    b = f(x - 3)
    return a() + b()


@cconv((10,))
def test_return_in_double_while(x):
    while x > 0:
        while x > 0:
            x = x - 1
            return x
    return -1


@cconv((3,))
def test_pow10(x):
    v = x
    j = 0
    while j < 3:
        i = 0
        while i < 3:
            v = v * x
            i = i + 1
        j = j + 1
    return v


@cconv((53,))
def test_closure_as_fv(x):
    def f():
        return x

    def g():
        return f()
    return g()
