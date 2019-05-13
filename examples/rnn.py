"""Example of an RNN in Myia.

Myia is still a work in progress, and this example may change in the future.
"""


import time
import numpy
from numpy.random import RandomState
from dataclasses import dataclass

from myia import myia, value_and_grad, ArithmeticData
from myia.prim.py_implementations import distribute, shape, tuple_setitem
# The following import installs custom tracebacks for inference errors
from myia.debug import traceback  # noqa


###########
# Options #
###########


device_type = 'cpu'
# device_type = 'cuda'  # Uncomment to run on the gpu


########
# Data #
########


# This just generates random data so we don't have to load a real dataset,
# but the model will work just as well on a real dataset.


def param(R, *size):
    """Generates a random array using the generator R."""
    return numpy.array(R.rand(*size) * 2 - 1, dtype='float32')


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


def rnn_parameters(*layer_sizes, seed=123123):
    """Generates parameters for a MLP given a list of layer sizes."""
    R = RandomState(seed=seed)
    parameters = []
    for i, o in zip(layer_sizes[:-1], layer_sizes[1:]):
        W = param(R, i, o)
        r = param(R, o, o)
        b = param(R, 1, o)
        h0 = param(R, 5, o)
        parameters.append((W, r, b, h0))
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
        shp = tuple_setitem(shape(self.h0), 0, shape(x[0])[0])
        h = distribute(self.h0, shp)
        # h = self.h0
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
def step(model, x, y):
    """Returns the loss and parameter gradients."""
    # value_and_grad will return cost(model, x, y) and dcost(...)/dmodel.
    # The 'model' argument can be omitted: by default the derivative wrt
    # the first argument is returned.
    return value_and_grad(cost, 'model')(model, x, y)


def run_helper(epochs, n, batch_size, layer_sizes):
    """Run a model with the specified layer sizes on n random batches.

    Arguments:
        epochs: How many epochs to run.
        n: Number of training batches to generate.
        batch_size: Number of samples per batch.
        layer_sizes: Sizes of the model's layers.
    """
    layers = []
    for W, R, b, h0 in rnn_parameters(*layer_sizes):
        layers.append(RNNLayer(W, R, b, h0))
        layers.append(Tanh())
    model = Sequential(tuple(layers))
    data = generate_data(n, batch_size, layer_sizes[0], layer_sizes[-1], 10)

    for _ in range(epochs):
        costs = []
        t0 = time.time()
        for inp, target in data:
            cost, dmodel = step(model, inp, target)
            costs.append(cost)
            model = model - (numpy.float32(0.01) * dmodel)
        c = sum(costs) / n
        t = time.time() - t0
        print(f'Cost: {c:15.10f}\tTime: {t:15.10f}')


# def test_run():
#     """Run the model.

#     This function is run automatically in the test suite to check against
#     regressions, so keep a low number of narrow layers to make sure it runs
#     quickly.
#     """
#     run_helper(1, 5, 5, (10, 3))


def run():
    """Run the model."""
    # run_helper(100, 10, 5, (10, 50, 1))
    run_helper(100, 10, 5, (10, 1))


if __name__ == '__main__':
    run()
