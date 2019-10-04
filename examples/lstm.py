"""Example of an RNN in Myia.

Myia is still a work in progress, and this example may change in the future.
"""


import time
from dataclasses import dataclass

import numpy
from numpy.random import RandomState

from myia import ArithmeticData, myia, value_and_grad
# The following import installs custom tracebacks for inference errors
from myia.debug import traceback  # noqa
from myia.xtype import Array

###########
# Options #
###########


dtype = 'float32'
device_type = 'cpu'
# device_type = 'cuda'  # Uncomment to run on the gpu


########
# Data #
########


# This just generates random data so we don't have to load a real dataset,
# but the model will work just as well on a real dataset.


def param(R, *size):
    """Generates a random array using the generator R."""
    return numpy.array(R.rand(*size) * 2 - 1, dtype=dtype)


def generate_data(n, batch_size, input_size, target_size, sequence_size,
                  *, seed=91):
    """Generate inputs and targets.

    Generates n batches of samples of size input_size, matched with
    a single target.
    """
    R = RandomState(seed=seed)
    return [([param(R, batch_size, input_size) for i in range(sequence_size)],
             param(R, batch_size, target_size))
            for i in range(n)]


def lstm_parameters(*layer_sizes, batch_size, seed=6666):
    """Generates parameters for a MLP given a list of layer sizes."""
    R = RandomState(seed=seed)
    i, h, *rest = layer_sizes

    W_i = param(R, i, h)
    W_f = param(R, i, h)
    W_c = param(R, i, h)
    W_o = param(R, i, h)

    R_i = param(R, h, h)
    R_f = param(R, h, h)
    R_c = param(R, h, h)
    R_o = param(R, h, h)

    b_i = param(R, 1, h)
    b_f = param(R, 1, h)
    b_c = param(R, 1, h)
    b_o = param(R, 1, h)

    s0 = numpy.zeros((1, h), dtype=dtype)
    c0 = numpy.zeros((1, h), dtype=dtype)

    parameters = [(
        W_i, W_f, W_c, W_o,
        R_i, R_f, R_c, R_o,
        b_i, b_f, b_c, b_o,
        s0, c0
    )]

    for i, o in zip((h, *rest[:-1]), rest):
        W = param(R, i, o)
        b = param(R, 1, o)
        parameters.append((W, b))

    return parameters


#########
# Model #
#########


# We generate an LSTM.


def sigmoid(x):
    """Sigmoid activation function."""
    return (numpy.tanh(x) + 1) / 2


@dataclass(frozen=True)
class Linear(ArithmeticData):
    """Linear layer."""

    W: 'Weights array'
    b: 'Biases vector'

    def apply(self, input):
        """Apply the layer."""
        return input @ self.W + self.b


@dataclass(frozen=True)
class Tanh(ArithmeticData):
    """Tanh layer."""

    def apply(self, input):
        """Apply the layer."""
        return numpy.tanh(input)


@dataclass(frozen=True)
class LSTMLayer(ArithmeticData):
    """LSTM layer."""

    W_i: Array
    W_f: Array
    W_c: Array
    W_o: Array

    R_i: Array
    R_f: Array
    R_c: Array
    R_o: Array

    b_i: Array
    b_f: Array
    b_c: Array
    b_o: Array

    s0: Array
    c0: Array

    def step(self, x_t, h_tm1, c_tm1):
        """Run one LSTM step."""
        i_t = sigmoid((x_t @ self.W_i) + (h_tm1 @ self.R_i) + self.b_i)
        f_t = sigmoid((x_t @ self.W_f) + (h_tm1 @ self.R_f) + self.b_f)
        o_t = sigmoid((x_t @ self.W_o) + (h_tm1 @ self.R_o) + self.b_o)

        c_hat_t = numpy.tanh(
            (x_t @ self.W_c) + (h_tm1 @ self.R_c) + self.b_c
        )

        c_t = f_t * c_tm1 + i_t * c_hat_t
        h_t = o_t * numpy.tanh(c_t)

        return h_t, c_t

    def apply(self, x):
        """Apply the layer."""
        s = self.s0
        c = self.c0
        for e in x:
            s, c = self.step(e, s, c)
        # Maybe collect and return the full list of s outputs?
        return s


@dataclass(frozen=True)
class Sequential(ArithmeticData):
    """Sequential layer, applies all sub-layers in order."""

    layers: 'Tuple of layers'

    def apply(self, x):
        """Apply the layer."""
        for layer in self.layers:
            x = layer.apply(x)
        return x


def cost(model, x, target):
    """Square difference loss."""
    y = model.apply(x)
    diff = target - y
    return sum(diff * diff)


# @myia(backend_options={'target': device_type})
@myia(backend='pytorch', backend_options={'device': device_type})
def step(model, lr, x, y):
    """Returns the loss and parameter gradients."""
    # value_and_grad will return cost(model, x, y) and dcost(...)/dmodel.
    # The 'model' argument can be omitted: by default the derivative wrt
    # the first argument is returned.
    _cost, dmodel = value_and_grad(cost, 'model')(model, x, y)
    return _cost, model - (lr * dmodel)


def run_helper(epochs, n, batch_size, layer_sizes):
    """Run a model with the specified layer sizes on n random batches.

    The first layer is an LSTM layer, the rest are linear+tanh.

    Arguments:
        epochs: How many epochs to run.
        n: Number of training batches to generate.
        batch_size: Number of samples per batch.
        layer_sizes: Sizes of the model's layers.
    """
    layers = []
    lstmp, *linp = lstm_parameters(*layer_sizes, batch_size=batch_size)
    layers.append(LSTMLayer(*lstmp))
    for W, b in linp:
        layers.append(Linear(W, b))
        layers.append(Tanh())
    model = Sequential(tuple(layers))
    data = generate_data(n, batch_size, layer_sizes[0], layer_sizes[-1], 10)
    lr = getattr(numpy, dtype)(0.01)

    for _ in range(epochs):
        costs = []
        t0 = time.time()
        for inp, target in data:
            cost, model = step(model, lr, inp, target)
            if isinstance(cost, numpy.ndarray):
                cost = float(cost)
            costs.append(cost)
        c = sum(costs) / n
        t = time.time() - t0
        print(f'Cost: {c:15.10f}\tTime: {t:15.10f}')


# We do not currently run this test in the test suite because it is too
# expensive to run.

# def test_run():
#     """Run the model.

#     This function is run automatically in the test suite to check against
#     regressions, so keep a low number of narrow layers to make sure it runs
#     quickly.
#     """
#     run_helper(1, 1, 5, (10, 3))


def run():
    """Run the model."""
    run_helper(100, 10, 5, (10, 7, 1))


if __name__ == '__main__':
    run()
