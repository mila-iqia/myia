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


def rnn_parameters(*layer_sizes, batch_size, seed=123123):
    """Generates parameters for an RNN given a list of layer sizes."""
    R = RandomState(seed=seed)
    i, h, *rest = layer_sizes

    W = param(R, i, h)
    U = param(R, h, h)
    b = param(R, 1, h)
    h0 = param(R, 1, h)
    parameters = [(W, U, b, h0)]

    for i, o in zip((h, *rest[:-1]), rest):
        W = param(R, i, o)
        b = param(R, 1, o)
        parameters.append((W, b))

    return parameters


#########
# Model #
#########


# We generate an RNN.


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
class RNNLayer(ArithmeticData):
    """RNN layer."""

    W: 'Input to state weights'
    R: 'State transition weights'
    b: 'Biases vector'
    h0: 'Initial state'

    def step(self, x, h_tm1):
        """Run one RNN step."""
        return numpy.tanh((x @ self.W) + (h_tm1 @ self.R) + self.b)

    def apply(self, x):
        """Apply the layer."""
        h = self.h0
        for e in x:
            h = self.step(e, h)
        # Maybe collect and return the full list of outputs?
        return h


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
    return _cost, model - (dmodel * lr)


def run_helper(epochs, n, batch_size, layer_sizes):
    """Run a model with the specified layer sizes on n random batches.

    The first layer is an RNN layer, the rest are linear+tanh.

    Arguments:
        epochs: How many epochs to run.
        n: Number of training batches to generate.
        batch_size: Number of samples per batch.
        layer_sizes: Sizes of the model's layers.
    """
    layers = []
    (W, U, b, h0), *linp = rnn_parameters(*layer_sizes, batch_size=batch_size)
    layers.append(RNNLayer(W, U, b, h0))
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


def test_run():
    """Run the model.

    This function is run automatically in the test suite to check against
    regressions, so keep a low number of narrow layers to make sure it runs
    quickly.
    """
    run_helper(1, 1, 5, (10, 3))


def run():
    """Run the model."""
    run_helper(100, 10, 5, (10, 7, 1))


if __name__ == '__main__':
    run()
