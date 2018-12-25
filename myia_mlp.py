from numpy import asscalar
from dataclasses import dataclass
from myia.optimizer.sgd import SGD

from myia.api import myia
from myia.dtype import *
from myia.composite import grad
from myia.prim.py_implementations import *


#########
# Model #
#########

# need to be done
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    e = np.exp(-2 * x)
    return (1 - e) / (1 + e)


# need to be done
def relu(x):
    e = np.exp(-2 * x)
    return (1 - e) / (1 + e)


def SigmoidLayer(inNum, outNum, init='random', dtype=np.float64):
    if init == 'zero':
        W = np.zeros((inNum, outNum)).astype(dtype)
    else:
        W = np.random.randn(inNum, outNum).astype(dtype)
    b = np.zeros((1, outNum)).astype(dtype)

    @dataclass()
    class SigmoidLayerBase:
        W: Array
        b: Array

        def apply(self, input):
            return sigmoid(input @ self.W + self.b)

        def update(self, opt, grads, moments):
            self.W, moments.W = opt(self.W, grads.W, moments.W)

    return SigmoidLayerBase(W, b)


def TanhLayer(inNum, outNum, init='random', dtype=np.float64):
    if init == 'zero':
        W = np.zeros((inNum, outNum)).astype(dtype)
    else:
        W = np.random.randn(inNum, outNum).astype(dtype)
    b = np.zeros((1, outNum)).astype(dtype)

    @dataclass()
    class TanhLayerBase:
        W: Array
        b: Array

        def apply(self, input):
            return tanh(input @ self.W + self.b)

        def update(self, opt, grads, moments):
            self.W, moments.W = opt(self.W, grads.W, moments.W)

    return TanhLayerBase(W, b)


def ReluLayer(inNum, outNum, init='random', dtype=np.float64):
    if init == 'zero':
        W = np.zeros((inNum, outNum)).astype(dtype)
    else:
        W = np.random.randn(inNum, outNum).astype(dtype)
    b = np.zeros((1, outNum)).astype(dtype)

    @dataclass()
    class ReluLayerBase:
        W: Array
        b: Array

        def apply(self, input):
            return relu(input @ self.W + self.b)

        def update(self, opt, grads, moments):
            self.W, moments.W = opt(self.W, grads.W, moments.W)

    return ReluLayerBase(W, b)


def SoftmaxLayer(inNum, outNum, init='random', dtype=np.float64):
    if init == 'zero':
        W = np.zeros((inNum, outNum)).astype(dtype)
    else:
        W = np.random.randn(inNum, outNum).astype(dtype)
    b = np.zeros((1, outNum)).astype(dtype)

    @dataclass()
    class SoftmaxLayerBase:
        W: Array
        b: Array

        def apply(self, input):
            out = input @ self.W + self.b
            div = numpy.exp(out)
            sm = array_reduce(scalar_add, div, ())
            sn = distribute(sm, (inNum, outNum))
            return div / sn

        def update(self, opt, grads, moments):
            self.W, moments.W = opt(self.W, grads.W, moments.W)

    return SoftmaxLayerBase(W, b)


def DenseLayer(inNum, outNum, init='random', dtype=np.float64):
    if init == 'zero':
        W = np.zeros((inNum, outNum)).astype(dtype)
    else:
        W = np.random.randn(inNum, outNum).astype(dtype)
    b = np.zeros((1, outNum)).astype(dtype)

    @dataclass()
    class DenseLayerBase:
        W: Array
        b: Array

        def apply(self, input):
            return input @ self.W + self.b

        def update(self, opt, grads, moments):
            self.W, moments.W = opt(self.W, grads.W, moments.W)

    return DenseLayerBase(W, b)

@dataclass(frozen=True)
class ConvLayer:
    W: Array

    def apply(self, input):
        a=array_conv(input, 5, 5, 2, 1)
        w=reshape(self.W, (4,1))
        xs = a @ w
        xs = reshape(xs, (4,4))
        return xs


@dataclass(frozen=True)
class Model:
    layers: Tuple

    def apply(self, x):
        for layer in self.layers:
            x = layer.apply(x)
        return x

    def update(self, opt, grads, moments):
        for layer, g, m in zip(self.layers, grads.layers, moments.layers):
            layer.update(opt, g, m)
        return self


#########
# Tests #
#########


def make_model(dtype=np.float64):
    return Model(
        layers=(
            DenseLayer(1, 1),
            #DenseLayer(2, 1),
            #LinearLayer(np.random.randn(1, 2).astype(dtype), np.zeros((1, 2)).astype(dtype)),
            #LinearLayer(np.random.randn(2, 1).astype(dtype), np.zeros((1, 1)).astype(dtype)),
        )
    )


def cost(model, x, y):
    yy = model.apply(x)
    diff = (yy - y)
    return asscalar(array_reduce(scalar_add, diff ** 2, ()))


def test_forward_specialize(model, x, y):
    return cost(model,x, y)


@myia
def test_forward_infer(model, x):
    return model.apply(x)


@myia
def test_backward_infer(model, x, y):
    return grad(cost)(model, x, y)

def run_model(n):

    model = make_model()
    moments = make_model()
    sgd = SGD()

    A = 2
    b = 0
    error = 0.1
    for i in range(n):
        # Let's make some data for a linear regression.
        N = 1
        X = np.random.randn(N, 1)

        # (noisy) Target values that we want to learn.
        t = A * X + b + np.random.randn(N,1) * error

        print('no.', i)
        print('run model x: ', X[0:N,:])
        print('run model y: ', t[0:N,:])
        loss = test_forward_specialize(model, X[0:N, :], t[0:N, :])
        print("loss:", loss)
        grads = test_backward_infer(model, X[0:N, :], t[0:N, :])
        sgd.update()

        print('model: ', model)
        print('grad: ', grads)
        print('moments: ', moments)

        model.update(sgd.step, grads, moments)
        print('update: ')
        print('model: ', model)
        print('moments: ', moments)
        print(' ')


if __name__ == '__main__':

    epoch = 100
    print('epoch: ', epoch)
    run_model(epoch)



