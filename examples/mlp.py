"""Example of an MLP in Myia.

Myia is still a work in progress, and this example may change in the future.
"""


import time
from dataclasses import dataclass

import numpy
from numpy.random import RandomState

from myia import ArithmeticData, myia, value_and_grad
from myia.api import to_device
from myia.debug import traceback  # noqa

###########
# Options #
###########


dtype = "float32"

backend = "pytorch"
# backend = 'relay'  # Uncomment to use relay backend

device_type = "cpu"
# device_type = 'cuda'  # Uncomment to run on the gpu

backend_options_dict = {
    "pytorch": {"device": device_type},
    "relay": {"target": device_type, "device_id": 0},
}

backend_options = backend_options_dict[backend]


###############
# Hyperparams #
###############


lr = getattr(numpy, dtype)(0.01)


########
# Data #
########


# This just generates random data so we don't have to load a real dataset,
# but the model will work just as well on a real dataset.


def param(R, *size):
    """Generates a random array using the generator R."""
    return numpy.array(R.rand(*size) * 2 - 1, dtype=dtype)


def generate_data(n, batch_size, input_size, target_size, *, seed=87):
    """Generate inputs and targets.

    Generates n batches of samples of size input_size, matched with
    a single target.
    """
    R = RandomState(seed=seed)
    return [
        (param(R, batch_size, input_size), param(R, batch_size, target_size))
        for i in range(n)
    ]


def mlp_parameters(*layer_sizes, seed=90909):
    """Generates parameters for a MLP given a list of layer sizes."""
    R = RandomState(seed=seed)
    parameters = []
    for i, o in zip(layer_sizes[:-1], layer_sizes[1:]):
        W = param(R, i, o)
        b = param(R, 1, o)
        parameters.append((W, b))
    return parameters


#########
# Model #
#########


# We generate a MLP model with some arbitrary number of layers and tanh
# activations.


@dataclass(frozen=True)
class Linear(ArithmeticData):
    """Linear layer."""

    W: "Weights array"
    b: "Biases vector"

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
class Sequential(ArithmeticData):
    """Sequential layer, applies all sub-layers in order."""

    layers: "Tuple of layers"

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


@myia(backend=backend, backend_options=backend_options, return_backend=True)
def step(model, x, y, lr):
    """Returns the loss and parameter gradients.

    value_and_grad will return cost(model, x, y) and dcost(...)/dmodel.
    The 'model' argument can be omitted: by default the derivative wrt
    the first argument is returned.
    """
    _cost, dmodel = value_and_grad(cost, "model")(model, x, y)
    return _cost, model - lr * dmodel


def run_helper(epochs, n, batch_size, layer_sizes):
    """Run a model with the specified layer sizes on n random batches.

    Arguments:
        epochs: How many epochs to run.
        n: Number of training batches to generate.
        batch_size: Number of samples per batch.
        layer_sizes: Sizes of the model's layers.
    """
    layers = []
    for W, b in mlp_parameters(*layer_sizes):
        layers.append(Linear(W, b))
        layers.append(Tanh())

    model = Sequential(tuple(layers))
    model = to_device(model, backend, backend_options)

    data = generate_data(n, batch_size, layer_sizes[0], layer_sizes[-1])

    for _ in range(epochs):
        costs = []
        t0 = time.time()
        for inp, target in data:
            cost, model = step(model, inp, target, lr)
            costs.append(cost)
        costs = [float(c.from_device()) for c in costs]
        c = sum(costs) / n
        t = time.time() - t0
        print(f"Cost: {c:15.10f}\tTime: {t:15.10f}")


def test_run():
    """Run the model.

    This function is run automatically in the test suite to check against
    regressions, so keep a low number of narrow layers to make sure it runs
    quickly.
    """
    run_helper(1, 5, 5, (10, 3))


def run():
    """Run the model."""
    run_helper(100, 10, 5, (10, 50, 1))


if __name__ == "__main__":
    run()
