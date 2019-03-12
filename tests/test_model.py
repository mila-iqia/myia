
import numpy
from dataclasses import dataclass
from numpy import ones as _ones, zeros as _zeros
from myia.dtype import Array, Tuple, pytype_to_myiatype
from myia.abstract import InferenceError
from myia.composite import grad
from myia.pipeline import standard_pipeline
from myia.prim.py_implementations import array_reduce, scalar_add

from .test_compile import parse_compare
from .test_grad import grad_test
from .test_infer import infer_std, af64_of, af32_of
from .common import MA, MB, MC, MD


MA = MA * 0.1
MB = MB * 0.1
MC = MC * 0.1
MD = MD * 0.1


#############
# Utilities #
#############


def ones(*shp, dtype='float64'):
    return _ones(shp, dtype=dtype)


def zeros(*shp, dtype='float64'):
    return _zeros(shp, dtype=dtype)


#########
# Model #
#########


def tanh(x):
    e = numpy.exp(-2 * x)
    return (1 - e) / (1 + e)


@dataclass(frozen=True)
class TanhLayer:
    W: Array
    b: Array

    def apply(self, input):
        return numpy.tanh(input @ self.W + self.b)


@dataclass(frozen=True)
class Model:
    layers: Tuple

    def apply(self, x):
        for layer in self.layers:
            x = layer.apply(x)
        return x


#########
# Tests #
#########


def make_model(dtype='float64'):
    return Model(
        layers=(
            TanhLayer(MA(6, 9, dtype=dtype), zeros(1, 9, dtype=dtype)),
            TanhLayer(MB(9, 10, dtype=dtype), zeros(1, 10, dtype=dtype)),
            TanhLayer(MC(10, 8, dtype=dtype), zeros(1, 8, dtype=dtype)),
        )
    )


Model_t = pytype_to_myiatype(Model, make_model())
Model_t_f32 = pytype_to_myiatype(Model, make_model('float32'))


def cost(model, x, y):
    yy = model.apply(x)
    diff = (yy - y)
    return (array_reduce(scalar_add, diff ** 2, ())).item()


@infer_std(
    (make_model(), MC(3, 6), af64_of(3, 8)),
    (make_model('float32'), MC(3, 6), InferenceError),
    (make_model('float32'), MC(3, 6, dtype='float32'), af32_of(3, 8)),
    (make_model(), MC(3, 9), InferenceError),
)
def test_forward_infer(model, x):
    return model.apply(x)


@parse_compare((make_model(), MC(3, 6)), array=True)
def test_forward_specialize(model, x):
    return model.apply(x)


@parse_compare((make_model(), MC(3, 6)), array=True, profile=True)
def test_forward_profile(model, x):
    return model.apply(x)


@infer_std(
    (make_model(), MC(3, 6), MC(3, 8), make_model()),
    (make_model(), MC(3, 6), MC(3, 9), InferenceError),
    (make_model('float32'), MC(3, 6), MC(3, 8), InferenceError),
    (make_model('float32'),
     MC(3, 6, dtype='float32'),
     MC(3, 8, dtype='float32'),
     make_model('float32')),
)
def test_backward_infer(model, x, y):
    return grad(cost)(model, x, y)


@grad_test((make_model(), MC(3, 6), MD(3, 8)),
           pipeline=standard_pipeline,
           rel_error=1e-1)
def test_backward_specialize(model, x, y):
    return cost(model, x, y)
