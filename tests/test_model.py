
import numpy
from dataclasses import dataclass
from numpy import ones as _ones, zeros as _zeros, asscalar
from myia.dtype import Array, Tuple, pytype_to_myiatype
from myia.infer import InferenceError
from myia.composite import grad
from myia.pipeline import standard_pipeline
from myia.prim.py_implementations import array_reduce, scalar_add
from myia.prim.shape_inferrers import ClassShape, TupleShape

from .test_compile import parse_compare
from .test_grad import grad_test
from .test_infer import infer_std
from .test_ainfer import infer_std as infer_std2
from .common import af32, af64, MA, MB, MC, MD


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
        return tanh(input @ self.W + self.b)


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
            TanhLayer(MA(6, 10, dtype=dtype), zeros(1, 10, dtype=dtype)),
            TanhLayer(MB(10, 8, dtype=dtype), zeros(1, 8, dtype=dtype)),
        )
    )


Model_t = pytype_to_myiatype(Model, make_model())
Model_t_f32 = pytype_to_myiatype(Model, make_model('float32'))


def cost(model, x, y):
    yy = model.apply(x)
    diff = (yy - y)
    return asscalar(array_reduce(scalar_add, diff ** 2, ()))


@infer_std(
    type=[
        ({'value': make_model()},
         {'value': MC(3, 6)},
         af64),
        ({'value': make_model('float32')},
         {'value': MC(3, 6)},
         InferenceError),
        ({'value': make_model('float32')},
         {'value': MC(3, 6, dtype='float32')},
         af32),
    ],
    shape=[
        ({'value': make_model()},
         {'value': MC(3, 6)},
         (3, 8)),
        ({'value': make_model()},
         {'value': MC(3, 9)},
         InferenceError),
    ]
)
def test_forward_infer(model, x):
    return model.apply(x)


@parse_compare((make_model(), MC(3, 6)), array=True)
def test_forward_specialize(model, x):
    return model.apply(x)


@infer_std(
    type=[
        ({'value': make_model()},
         {'value': MC(3, 6)},
         {'value': MD(3, 8)},
         Model_t),
        ({'value': make_model('float32')},
         {'value': MC(3, 6)},
         {'value': MD(3, 8)},
         InferenceError),
        ({'value': make_model('float32')},
         {'value': MC(3, 6, dtype='float32')},
         {'value': MD(3, 8, dtype='float32')},
         Model_t_f32),
    ],
    shape=[
        ({'value': make_model()},
         {'value': MC(3, 6)},
         {'value': MD(3, 8)},
         ClassShape({
             'layers': TupleShape((ClassShape({'W': (6, 10), 'b': (1, 10)}),
                                   ClassShape({'W': (10, 8), 'b': (1, 8)})))})
         ),
        ({'value': make_model()},
         {'value': MC(3, 9)},
         {'value': MC(3, 8)},
         InferenceError),
    ]
)
def test_backward_infer(model, x, y):
    return grad(cost)(model, x, y)


@grad_test((make_model(), MC(3, 6), MD(3, 8)),
           pipeline=standard_pipeline,
           rel_error=1e-1)
def test_backward_specialize(model, x, y):
    return cost(model, x, y)


@infer_std2(
    type=[
        ({'value': make_model()},
         {'value': MC(3, 6)},
         {'value': MD(3, 8)},
         Model_t),
        ({'value': make_model('float32')},
         {'value': MC(3, 6)},
         {'value': MD(3, 8)},
         InferenceError),
        ({'value': make_model('float32')},
         {'value': MC(3, 6, dtype='float32')},
         {'value': MD(3, 8, dtype='float32')},
         Model_t_f32),
    ],
    shape=[
        ({'value': make_model()},
         {'value': MC(3, 6)},
         {'value': MD(3, 8)},
         ClassShape({
             'layers': TupleShape((ClassShape({'W': (6, 10), 'b': (1, 10)}),
                                   ClassShape({'W': (10, 8), 'b': (1, 8)})))})
         ),
        ({'value': make_model()},
         {'value': MC(3, 9)},
         {'value': MC(3, 8)},
         InferenceError),
    ]
)
def test_backward_infer_2(model, x, y):
    return grad(cost)(model, x, y)
