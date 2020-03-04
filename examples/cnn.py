"""Example of a CNN in Myia.

Myia is still a work in progress, and this example may change in the future.
"""
from __future__ import print_function

import time
from dataclasses import dataclass

import numpy
import torch
from numpy.random import RandomState
from torchvision import datasets, transforms

from myia import ArithmeticData, myia
from myia.api import to_device
# The following import installs custom tracebacks for inference errors
from myia.debug import traceback  # noqa
# TODO: add back value_and_grad
from myia.frontends import activate_frontend
from myia.public_api import conv2d, log_softmax, max_pool2d, nll_loss, reshape

# TODO: add back nll_loss

activate_frontend('pytorch')


###########
# Options #
###########


dtype = 'float32'

backend = 'pytorch'
# backend = 'relay'  # Uncomment to use relay backend

device_type = 'cpu'
# device_type = 'cuda'  # Uncomment to run on the gpu

backend_options_dict = {
    'pytorch': {'device': device_type},
    'relay': {'target': device_type, 'device_id': 0}
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


def cnn_parameters(in_channels, out_channels, kernel_size, stride=1,
                   padding=0, dilation=1, groups=1,
                   use_bias=True, padding_mode='zeros', seed=90909):
    """Generates parameters for a cnn layer given a list of layer sizes."""
    R = RandomState(seed=seed)
    transposed = False
    if transposed:
        weight = param(R, in_channels, out_channels // groups, *kernel_size)
    else:
        weight = param(R, out_channels, in_channels // groups, *kernel_size)
    bias = param(R, out_channels)
    return [(weight, bias)]


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
class Conv2d(ArithmeticData):
    """Conv2d layer."""

    W: 'Weights array'
    b: 'Biases vector'
    stride: 'stride' = (1, 1)
    padding: 'padding' = (0, 0)
    dilation: 'dilation' = (1, 1)
    groups: 'groups' = 1

    def apply(self, input):
        """Apply the layer."""
        return conv2d(input, self.W, self.b, self.stride,
                      self.padding, self.dilation, self.groups)


@dataclass(frozen=True)
class Linear(ArithmeticData):
    """Linear layer."""

    W: 'Weights array'
    b: 'Biases vector'

    def apply(self, input):
        """Apply the layer."""
        return input @ self.W + self.b


@dataclass(frozen=True)
class Reshape(ArithmeticData):
    """Reshape layer."""

    dims: 'Dimensions tuple'

    def apply(self, input):
        """Apply the layer."""
        return reshape(input, self.dims)


@dataclass(frozen=True)
class Tanh(ArithmeticData):
    """Tanh layer."""

    def apply(self, input):
        """Apply the layer."""
        return numpy.tanh(input)


@dataclass(frozen=True)
class MaxPool2d(ArithmeticData):
    """MaxPool2d layer."""

    kernel_size: "Kernel_size tuple"
    stride: "Stride tuple"
    padding: "padding" = (0, 0)
    dilation: "dilation" = (1, 1)

    def apply(self, input):
        """Apply the layer."""

        return max_pool2d(input, self.kernel_size, self.stride,
                          self.padding, self.dilation)


@dataclass(frozen=True)
class LogSoftmax(ArithmeticData):
    """LogSoftmax layer."""

    # dims: Static(int)
    dims: "Dimensions tuple"

    def apply(self, input):
        """Apply the layer."""
        return log_softmax(input, dim=self.dims)


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
    output = model.apply(x)
    return nll_loss(output, target)


@myia(backend=backend, backend_options=backend_options, return_backend=True,
      specialize_values=["model"])
def step(model, x, y, lr):
    """Returns the loss and parameter gradients.

    value_and_grad will return cost(model, x, y) and dcost(...)/dmodel.
    The 'model' argument can be omitted: by default the derivative wrt
    the first argument is returned.
    """

    """
    _cost, dmodel = value_and_grad(cost, 'model')(model, x, y)
    return _cost, model - lr * dmodel
    # """
    return cost(model, x, y), cost(model, x, y)


def run_helper(epochs, n, batch_size, layer_sizes):
    """Run a model with the specified layer sizes on n random batches.

    Arguments:
        epochs: How many epochs to run.
        n: Number of training batches to generate.
        batch_size: Number of samples per batch.
        layer_sizes: Sizes of the model's layers.
    """
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(123)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    cnn1 = cnn_parameters(1, 20, (5, 1))
    cnn2 = cnn_parameters(20, 50, (5, 1))
    mlp = mlp_parameters(*layer_sizes)
    model = Sequential((
        Conv2d(cnn1[0][0], cnn1[0][1]),
        Tanh(),
        MaxPool2d((2, 2), (2, 2), (0, 0), (1, 1)),
        Conv2d(cnn2[0][0], cnn2[0][1]),
        Tanh(),
        MaxPool2d((2, 2), (2, 2), (0, 0), (1, 1)),
        Reshape((-1, 1400)),
        # Reshape((-1, 800)),
        Linear(mlp[0][0], mlp[0][1]),
        Tanh(),
        Linear(mlp[1][0], mlp[1][1]),
        LogSoftmax(1)
    ))

    model = to_device(model, backend, backend_options, broaden=False)

    for _ in range(epochs):
        costs = []
        t0 = time.time()
        for batch_idx, (inp, target) in enumerate(train_loader):
            inp = inp.numpy()
            target = target.float().numpy()
            # cost, model = step(model, inp, target, lr)
            cost, _ = step(model, inp, target, lr)
            print(batch_idx)
            costs.append(cost)
        costs = [float(c.from_device()) for c in costs]
        c = sum(costs) / n
        t = time.time() - t0
        print(f'Cost: {c:15.10f}\tTime: {t:15.10f}')


def run():
    """Run the model."""
    # run_helper(100, 10, 5, (800, 50, 10))
    run_helper(100, 10, 5, (1400, 50, 10))


if __name__ == '__main__':
    run()
