import numpy as np
from dataclasses import dataclass
import pytest

from myia.api import myia, to_device
from myia.cconv import closure_convert
from myia.dtype import Bool, EnvType
from myia.abstract import InferenceError
from myia.ir import clone
from myia.pipeline import \
    scalar_parse as parse, scalar_debug_compile as compile
from myia.pipeline.steps import convert_arg, convert_result, NumpyChecker
from myia.prim.py_implementations import tuple_getitem
from myia.utils import newenv
from myia.abstract import ArrayWrapper
from myia.compile import load_backend

from .common import Point, Point3D, i64, f64, to_abstract_test, ai64_of, \
    ai32_of, af64_of


def test_myia():
    @myia
    def f(x, y):
        return x + y

    assert f(10, 20) == 30

    fi = f.compile((10, 20))
    ff = f.compile((10.0, 20.0))
    assert fi is not ff
    assert fi is f.compile((100, 200))

    with pytest.raises(InferenceError):
        f(10)
    with pytest.raises(InferenceError):
        f(10, 20, 30)
    with pytest.raises(InferenceError):
        f((10, 20), (30, 40))


def test_myia_specialize_values():
    @myia(specialize_values=['c'])
    def f(c, x, y):
        if c:
            return x + y
        else:
            return x * y

    assert f(True, 10, 20) == 30
    assert f(False, 10, 20) == 200

    ft = f.compile((True, 10, 20))
    ff = f.compile((False, 10, 20))
    assert ft is not ff


def test_myia_return_backend():
    @myia(return_backend=True)
    def f(x, y):
        return x + y

    # Only tests that it returns something and it works
    assert f(10, 20) is not None


def test_myia_struct_arg():
    @myia
    def f(pt):
        return pt.x

    x = f(Point(5, 6))
    assert x == 5


def test_myia_return_struct():
    @myia
    def f(x, y):
        return Point(x, y)

    pt = f(5, 6)
    assert pt == Point(5, 6)


def test_convert_arg():

    backend = NumpyChecker()

    def _convert(data, typ):
        return convert_arg(data, to_abstract_test(typ), backend)

    # Leaves

    assert _convert(True, Bool) is True
    assert _convert(False, Bool) is False
    assert _convert(10, i64) == 10
    assert _convert(1.5, f64) == 1.5
    with pytest.raises(TypeError):
        _convert(newenv, EnvType)

    # Class -> Tuple conversion

    pt = Point(1, 2)
    pt3 = Point3D(1, 2, 3)
    assert list(_convert(pt, Point(i64, i64))) == [1, 2]
    with pytest.raises(TypeError):
        _convert((1, 2), Point(i64, i64))

    assert list(_convert((pt, pt),
                (Point(i64, i64), Point(i64, i64)))) == [(1, 2), (1, 2)]

    assert _convert([pt, pt, pt],
                    [Point(i64, i64)]) == [(1, 2), (1, 2), (1, 2)]

    # Arrays

    fmat = np.ones((5, 8))
    imat = np.ones((5, 8), dtype='int32')

    assert _convert(fmat, af64_of(5, 8)) is fmat
    assert _convert(imat, ai32_of(5, 8)) is imat
    with pytest.raises(TypeError):
        _convert(imat, ai64_of(5, 8))

    # Misc errors

    with pytest.raises(TypeError):
        _convert(10, f64)
    with pytest.raises(TypeError):
        _convert("blah", to_abstract_test("blah"))
    with pytest.raises(TypeError):
        _convert(1.5, i64)
    with pytest.raises(TypeError):
        _convert(10, (i64, i64))
    with pytest.raises(TypeError):
        _convert((1,), (i64, i64))
    with pytest.raises(TypeError):
        _convert((1, 2, 3), (i64, i64))
    with pytest.raises(TypeError):
        _convert((1, 2, 3), [i64])
    with pytest.raises(TypeError):
        _convert(pt3, Point(i64, i64))
    with pytest.raises(TypeError):
        _convert(10, ai64_of())
    with pytest.raises(TypeError):
        _convert(10, ai64_of())
    with pytest.raises(TypeError):
        _convert(1, Bool)


@pytest.fixture(params=[
    pytest.param(True),
    pytest.param(False)
    ])
def _return_backend(request):
    return request.param


def test_convert_result(_return_backend):

    backend = NumpyChecker()

    def _convert(data, typ1, typ2):
        return convert_result(data,
                              to_abstract_test(typ1),
                              to_abstract_test(typ2),
                              backend,
                              _return_backend)

    # Leaves

    assert _convert(True, Bool, Bool) is True
    assert _convert(False, Bool, Bool) is False
    assert _convert(10, i64, i64) == 10
    assert _convert(1.5, f64, f64) == 1.5

    # Tuple -> Class conversion

    pt = Point(1, 2)
    assert _convert(pt, Point(i64, i64), Point(i64, i64)) == pt
    assert _convert((1, 2), Point(i64, i64), (i64, i64)) == pt

    assert _convert(((1, 2), (1, 2)),
                    (Point(i64, i64), Point(i64, i64)),
                    ((i64, i64), (i64, i64))) == \
        (pt, pt)

    assert _convert([(1, 2), (1, 2), (1, 2)],
                    [Point(i64, i64)],
                    [(i64, i64)]) == \
        [pt, pt, pt]


def test_function_arg():
    """Give a Python function as an argument."""
    def square(x):  # pragma: no cover
        return x * x

    @compile
    def f(fn, x, y):
        return fn(x + y)

    assert f(square, 10, 5) == 225


def test_function_in_tuple():
    """Give a tuple of functions as an argument."""
    def square(x):  # pragma: no cover
        return x * x

    def double(x):  # pragma: no cover
        return x + x

    @compile
    def f(fns, x, y):
        f0, f1 = fns
        return f1(f0(x + y))

    assert f((square, double), 10, 5) == 450
    assert f((double, square), 10, 5) == 900


def test_return_closure():
    """Return a closure."""
    @compile
    def f(x, y):
        def g():
            return x + y
        return g

    assert f(4, 5)() == 9


def test_return_closure_partial():
    """Return a closure (after closure conversion)."""
    @parse
    def f(x, y):
        def g():
            return x + y
        return g

    f = clone(closure_convert(f))
    f = compile(f)

    g = f(4, 5)
    assert g() == 9


def test_return_closure_tuple():
    """Return a tuple of closures."""
    @compile
    def f(x, y):
        def g():
            return x + y

        def h():
            return x * y
        return (g, h)

    g, h = f(4, 5)
    assert g() == 9
    assert h() == 20


def test_refeed():
    """Return a closure, then use the closure as an argument."""
    @compile
    def f(fn, x, y):
        def g():
            return x + y
        if x == 0:
            return g
        else:
            return fn()

    g = f(None, 0, 6)
    assert g() == 6
    assert f(g, 10, 20) == 6


def test_return_primitive():
    """Return a primitive."""
    @compile
    def f():
        return tuple_getitem

    g = f()
    assert g((1, 2, 3), 0) == 1


def test_return_graph():
    @compile
    def f():
        def g():
            return 42
        return g

    g = f()
    assert g() == 42


def test_bad_call1():
    @compile
    def f():
        return 42

    with pytest.raises(RuntimeError):
        f(1)


def test_bad_call2():
    @compile
    def f():
        def g():
            return 42
        return g(0)

    with pytest.raises(RuntimeError):
        f()


def test_tail_call():
    @compile
    def f():
        x = 1000
        while x > 0:
            x = x - 1
        return x


#####################################################
# Test convert_arg_init functions used by to_device #
#####################################################

device_type = 'cpu'
# device_type = 'cuda'  # Uncomment to run on the gpu


@pytest.fixture(params=[
    pytest.param(('nnvm', {'target': 'cpu', 'device_id': 0}), id='nnvm-cpu'),
    pytest.param(('relay', {'target': 'cpu', 'device_id': 0}), id='relay-cpu'),
    pytest.param(('pytorch', {'device': 'cpu'}), id='pytorch-cpu')])
def backend_opt(request):
    name, options = request.param
    return (name, options)


__W = np.array(
    [
        [0.22490819, -0.71516967, -0.6626606],
        [-0.14859799, 0.924496, 0.4032511]
    ]
)

__b = np.array([[-0.00300545, 0.10826713, -0.29246148]])


def test__convert_arg_init_AbstractTuple(backend_opt):
    backend, backend_options = backend_opt
    b = load_backend(backend, backend_options)

    model = (9, 5.0)
    m = to_device(model, backend, backend_options)

    assert isinstance(m, tuple)
    assert isinstance(b.to_scalar(m[0]), int)
    assert isinstance(b.to_scalar(m[1]), float)
    assert b.to_scalar(m[0]) == 9
    assert b.to_scalar(m[1]) == 5.0


def test__convert_arg_init_AbstractList(backend_opt):
    backend, backend_options = backend_opt
    b = load_backend(backend, backend_options)

    model = [91.0, 51.0]
    m = to_device(model, backend, backend_options)

    assert isinstance(m, list)
    assert isinstance(b.to_scalar(m[0]), float)
    assert isinstance(b.to_scalar(m[1]), float)
    assert b.to_scalar(m[0]) == 91.0
    assert b.to_scalar(m[1]) == 51.0


def test__convert_arg_init_AbstractClass(backend_opt):
    backend, backend_options = backend_opt
    b = load_backend(backend, backend_options)

    @dataclass(frozen=True)
    class A():

        s: 'scalar number'

        def apply(self, input):
            """Apply the layer."""
            return (input, self.s)

    model = A(2.0)
    m = to_device(model, backend, backend_options)

    assert isinstance(m, A)
    assert isinstance(b.to_scalar(m.s), float)
    assert b.to_scalar(m.s) == 2.0


def test__convert_arg_init_AbstractArray(backend_opt):
    backend, backend_options = backend_opt
    b = load_backend(backend, backend_options)
    m = to_device(__W, backend, backend_options)

    assert isinstance(m, ArrayWrapper)
    np.testing.assert_allclose(b.to_numpy(m.array), __W)


def test__convert_arg_init_AbstractUnion(backend_opt):
    # TODO
    pass


def test__convert_arg_init_AbstractScalar(backend_opt):
    backend, backend_options = backend_opt
    b = load_backend(backend, backend_options)

    model1 = 3.0
    m1 = to_device(model1, backend, backend_options)
    assert isinstance(b.to_scalar(m1), float)
    assert b.to_scalar(m1) == 3.0

    model2 = 14
    m2 = to_device(model2, backend, backend_options)
    assert isinstance(b.to_scalar(m2), int)
    assert b.to_scalar(m2) == 14

#####################################################
