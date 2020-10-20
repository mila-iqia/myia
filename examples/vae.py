"""Example of an MLP in Myia.

Myia is still a work in progress, and this example may change in the future.
"""

import time
from dataclasses import dataclass

import numpy
import torch
from numpy.random import RandomState
from torchvision import datasets, transforms

import myia.public_api as pub
from myia import ArithmeticData, myia, value_and_grad
from myia.api import to_device
from myia.debug import traceback  # noqa
from myia.operations import array_exp, array_pow, random_initialize

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


@dataclass(frozen=True)
class VAE(ArithmeticData):
    """Sequential layer, applies all sub-layers in order."""

    fc1: "layer fc1"
    fc21: "layer fc21"
    fc22: "layer fc22"
    fc3: "layer fc3"
    fc4: "layer fc4"

    def encode(self, x):
        h1 = pub.relu(self.fc1.apply(x))
        return self.fc21.apply(h1), self.fc22.apply(h1)

    def reparameterize(self, mu, logvar, rstate):
        std = array_exp(0.5 * logvar)
        eps, rstate = pub.uniform(rstate, (2, 20), -1.0, 1.0)
        return mu + eps * std, rstate

    def decode(self, z):
        h3 = pub.relu(self.fc3.apply(z))
        return pub.sigmoid(self.fc4.apply(h3))

    def forward(self, x, rstate):
        mu, logvar = self.encode(pub.reshape(x, (-1, 784)))
        z, rstate = self.reparameterize(mu, logvar, rstate)
        return self.decode(z), mu, logvar, rstate


params = (
    mlp_parameters(*(784, 400))[0],
    mlp_parameters(*(400, 20))[0],
    mlp_parameters(*(400, 20))[0],
    mlp_parameters(*(20, 400))[0],
    mlp_parameters(*(400, 784))[0],
)

model = VAE(
    Linear(params[0][0], params[0][1]),
    Linear(params[1][0], params[1][1]),
    Linear(params[2][0], params[2][1]),
    Linear(params[3][0], params[3][1]),
    Linear(params[4][0], params[4][1]),
)

model = to_device(model, backend, backend_options, broaden=False)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = pub.binary_cross_entropy(
        recon_x, pub.reshape(x, (-1, 784)), reduction="sum"
    )

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * pub._sum(1 + logvar - array_pow(mu, 2) - array_exp(logvar))

    return BCE + KLD


def cost(model, data, rstate):
    recon_batch, mu, logvar, _rstate = model.forward(data, rstate)
    loss = loss_function(recon_batch, data, mu, logvar)
    return loss.item(), _rstate


@myia(backend=backend, backend_options=backend_options, return_backend=True)
def step(model, data, lr, rstate):
    """Returns the loss and parameter gradients.

    value_and_grad will return cost(model, x, y) and dcost(...)/dmodel.
    The 'model' argument can be omitted: by default the derivative wrt
    the first argument is returned.
    """
    (_cost, rstate), dmodel = value_and_grad(cost, "model")(
        model, data, rstate, dout=(1, random_initialize(0))
    )
    return _cost, model - lr * dmodel, rstate


@myia(backend=backend, backend_options=backend_options, return_backend=True)
def step_eval(model, data, rstate):
    """Returns the loss and parameter gradients.

    value_and_grad will return cost(model, x, y) and dcost(...)/dmodel.
    The 'model' argument can be omitted: by default the derivative wrt
    the first argument is returned.
    """
    return cost(model, data, rstate)


@myia(backend=backend, backend_options=backend_options, return_backend=True)
def step_init_seed():
    """Returns the loss and parameter gradients.

    value_and_grad will return cost(model, x, y) and dcost(...)/dmodel.
    The 'model' argument can be omitted: by default the derivative wrt
    the first argument is returned.
    """
    return random_initialize(1)


lr = getattr(numpy, dtype)(0.01)

if __name__ == "__main__":
    seed = 123
    cuda = False
    batch_size = 2
    epochs = 1

    torch.manual_seed(seed)

    device = torch.device("cuda" if cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    rand_state = step_init_seed()

    for _ in range(epochs):
        costs = []
        t0 = time.time()
        for i, (data, _) in enumerate(train_loader):
            print("i", i + 1, "/", len(train_loader))
            _cost, model, rand_state = step(
                model, data.reshape((batch_size, 784)).numpy(), lr, rand_state
            )
            costs.append(_cost)
        costs = [float(c.from_device()) for c in costs]
        c = sum(costs) / len(costs)
        t = time.time() - t0
        print(f"Cost: {c:15.10f}\tTime: {t:15.10f}")

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    costs = []
    t0 = time.time()
    for i, (data, _) in enumerate(test_loader):
        _cost, rand_state = step_eval(
            model, data.reshape((batch_size, 784)).numpy(), rand_state
        )
        costs.append(_cost)
    costs = [float(c.from_device()) for c in costs]
    c = sum(costs) / len(costs)
    t = time.time() - t0
    print(f"Cost: {c:15.10f}\tTime: {t:15.10f}")
