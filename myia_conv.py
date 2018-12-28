from numpy import asscalar
from dataclasses import dataclass
from myia.optimizer.sgd import SGD

from myia.api import myia
from myia.dtype import Float, Tuple, Array
from myia.composite import grad, cast
from myia.prim.py_implementations import *


#########
# Model #
#########


def between(beg, end, mid):
    # 判断mid是否位于begin和end之间
    return end > mid >= beg or end < mid <= beg


def get_slice(a, beg, end, delta=1):
    # 数组切片get方式
    if delta == 0:
        raise ValueError("slice step cannot be 0")
    # 将负数下标转化一下
    if beg < 0: beg += len(a)
    if end < 0: end += len(a)
    # 如果转化完成之后依然不在合法范围内，则返回空列表
    if beg < 0 and end < 0 or beg >= len(a) and end >= len(a):
        return []
    # 如果方向不同，则返回空列表
    if (end - beg) * delta <= 0:
        return []
    # 将越界的部分进行裁剪
    beg = max(0, min(beg, len(a) - 1))
    end = max(-1, min(end, len(a)))
    ans = []
    i = beg
    while between(beg, end, i):
        ans.append(a[i])
        i += delta
    return ans


# need to be done
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    e = np.exp(-2 * x)
    return (1 - e) / (1 + e)

def relu(x):
    return cast(x, typeof(1.0))

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

@dataclass()
class ReluLayer:
    def apply(self, input):
        return relu(input)

    def update(self, opt, grads, moments):
        pass

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
            # need to be done
            sn = distribute(sm, (2, 1))
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
            DenseLayer(1, 2),
            ReluLayer(),
            DenseLayer(2, 1)
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
    # Let's make some data for a linear regression.
    N = 1
    X = np.random.randn(N, 1)

    # (noisy) Target values that we want to learn.
    t = A * X + b + np.random.randn(N,1) * error
    for i in range(n):
        loss = test_forward_specialize(model, X[0:N, :], t[0:N, :])
        print('no.', i, "loss:", loss)
        grads = test_backward_infer(model, X[0:N, :], t[0:N, :])
        sgd.update()
        model.update(sgd.step, grads, moments)
    print('model: ', model)


if __name__ == '__main__':

    epoch = 100
    print('epoch: ', epoch)

    run_model(epoch)

