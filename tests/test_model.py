import typing
from dataclasses import dataclass

import numpy
from numpy import ones as _ones, zeros as _zeros

from myia.abstract import from_value
from myia.operations import array_reduce, grad, scalar_add
from myia.pipeline import standard_pipeline
from myia.testing.common import MA, MB, MC, MD, af32_of, af64_of
from myia.testing.multitest import mt, run
from myia.utils import InferenceError
from myia.xtype import Array

from .test_grad import gradient
from .test_infer import infer_standard

MA = MA * 0.1
MB = MB * 0.1
MC = MC * 0.1
MD = MD * 0.1


#############
# Utilities #
#############


def ones(*shp, dtype="float64"):
    return _ones(shp, dtype=dtype)


def zeros(*shp, dtype="float64"):
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
    layers: typing.Tuple

    def apply(self, x):
        for layer in self.layers:
            x = layer.apply(x)
        return x


#########
# Tests #
#########


def make_model(dtype="float64"):
    return Model(
        layers=(
            TanhLayer(MA(6, 9, dtype=dtype), zeros(1, 9, dtype=dtype)),
            TanhLayer(MB(9, 10, dtype=dtype), zeros(1, 10, dtype=dtype)),
            TanhLayer(MC(10, 8, dtype=dtype), zeros(1, 8, dtype=dtype)),
        )
    )


Model_t = from_value(make_model(), broaden=True)
Model_t_f32 = from_value(make_model("float32"), broaden=True)


def cost(model, x, y):
    yy = model.apply(x)
    diff = yy - y
    return (array_reduce(scalar_add, diff ** 2, ())).item()


@mt(
    infer_standard(make_model(), MC(3, 6), result=af64_of(3, 8)),
    infer_standard(make_model("float32"), MC(3, 6), result=InferenceError),
    infer_standard(
        make_model("float32"), MC(3, 6, dtype="float32"), result=af32_of(3, 8)
    ),
    infer_standard(make_model(), MC(3, 9), result=InferenceError),
    run(make_model(), MC(3, 6)),
)
def test_forward(model, x):
    return model.apply(x)


@mt(
    infer_standard(make_model(), MC(3, 6), MC(3, 8), result=make_model()),
    infer_standard(make_model(), MC(3, 6), MC(3, 9), result=InferenceError),
    infer_standard(
        make_model("float32"), MC(3, 6), MC(3, 8), result=InferenceError
    ),
    infer_standard(
        make_model("float32"),
        MC(3, 6, dtype="float32"),
        MC(3, 8, dtype="float32"),
        result=make_model("float32"),
    ),
)
def test_backward_infer(model, x, y):
    return grad(cost)(model, x, y)


@gradient(
    make_model(), MC(3, 6), MD(3, 8), pipeline=standard_pipeline, rel_error=1e-1
)
def test_backward_specialize(model, x, y):
    return cost(model, x, y)
