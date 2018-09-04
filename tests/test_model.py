
import numpy
from dataclasses import dataclass
from numpy import cast, ones as _ones, zeros as _zeros
from numpy.random import rand as _rand
from myia.dtype import Array, Tuple
from myia.infer import InferenceError

from .test_infer import infer_std
from .test_specialize import specialize_std
from .common import af32, af64


#############
# Utilities #
#############


def ones(*shp, dtype='float64'):
    return _ones(shp, dtype=dtype)


def zeros(*shp, dtype='float64'):
    return _zeros(shp, dtype=dtype)


def rand(*shp, dtype='float64'):
    return cast[dtype](_rand(*shp) - 0.5)


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


inp = rand(3, 10)


def make_model(dtype='float64'):
    return Model(
        layers=(
            TanhLayer(rand(10, 15, dtype=dtype), zeros(1, 15, dtype=dtype)),
            TanhLayer(rand(15, 8, dtype=dtype), zeros(1, 8, dtype=dtype)),
        )
    )


@infer_std(
    type=[
        ({'value': make_model()},
         {'value': rand(3, 10)},
         af64),
        ({'value': make_model('float32')},
         {'value': rand(3, 10)},
         InferenceError),
        ({'value': make_model('float32')},
         {'value': rand(3, 10, dtype='float32')},
         af32),
    ],
    shape=[
        ({'value': make_model()},
         {'value': rand(3, 10)},
         (3, 8)),
        ({'value': make_model()},
         {'value': rand(3, 14)},
         InferenceError),
    ]
)
def test_forward_infer(model, x):
    return model.apply(x)


@specialize_std((make_model(), rand(3, 10)))
def test_forward_specialize(model, x):
    return model.apply(x)
