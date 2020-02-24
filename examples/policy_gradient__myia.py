"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from myia import myia, value_and_grad
from myia.ir import sexp_to_node
from myia.lib import setter_from_getter
from myia.abstract import Macro, build_value, macro, myia_static
from myia.frontends import activate_frontend

from myia.debug import traceback

activate_frontend('pytorch')

from myia.operations import primitives as P
from dataclasses import dataclass
import numpy as np

import gym


import time
from dataclasses import dataclass

import numpy
from numpy.random import RandomState

from myia import ArithmeticData, myia, value_and_grad
from myia.api import to_device
# The following import installs custom tracebacks for inference errors
from myia.debug import traceback  # noqa
"""
# """
from __future__ import print_function
import torch

from myia import ArithmeticData, myia, value_and_grad
from myia.frontends import activate_frontend

from myia.debug import traceback

from dataclasses import dataclass

# import time

import gym
import numpy as np
from numpy.random import RandomState

from myia.api import to_device
# The following import installs custom tracebacks for inference errors
from myia.debug import traceback  # noqa
# """

from myia.public_api import softmax, _sum, mean, std

activate_frontend('pytorch')


dtype = 'float32'
backend = 'pytorch'
backend_options = {'device': 'cpu'}

HIDDEN_LAYER = 24  # NN hidden layer size
lr = LR = 0.01
GAMMA = 0.99

INPUT_SIZE = 4
OUTPUT_SIZE = 2

ENV = gym.make('CartPole-v0').unwrapped
HISTORY = []


def param(R, *size):
    """Generates a random array using the generator R."""
    return np.array(R.rand(*size) * 2 - 1, dtype=dtype)


def generate_data(n, batch_size, input_size, target_size, *, seed=87):
    """Generate inputs and targets.

    Generates n batches of samples of size input_size, matched with
    a single target.
    """
    R = RandomState(seed=seed)
    return [(param(R, batch_size, input_size),
             param(R, batch_size, target_size))
            for i in range(n)]


def mlp_parameters(*layer_sizes, seed=90909):
    """Generates parameters for a MLP given a list of layer sizes."""
    R = RandomState(seed=seed)
    parameters = []
    for i, o in zip(layer_sizes[:-1], layer_sizes[1:]):
        W = param(R, i, o)
        b = param(R, 1, o)
        parameters.append((W, b))
    return parameters


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
        return np.tanh(input)


@dataclass(frozen=True)
class Softmax(ArithmeticData):
    """LogSoftmax layer."""

    # dims: Static(int)
    dims: "Dimensions tuple"

    def apply(self, input):
        """Apply the layer."""
        return softmax(input, dim=self.dims)


@dataclass(frozen=True)
class Sequential(ArithmeticData):
    """Sequential layer, applies all sub-layers in order."""

    layers: 'Tuple of layers'

    def apply(self, x):
        """Apply the layer."""
        for layer in self.layers:
            x = layer.apply(x)
        return x


layer_sizes = (4, 24, 2)

mlp = mlp_parameters(*layer_sizes)
model = Sequential((
    Linear(mlp[0][0], mlp[0][1]),
    Tanh(),
    Linear(mlp[1][0], mlp[1][1]),
    Softmax(1)
))

model = to_device(model, backend, backend_options, broaden=False)


use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

@myia(backend=backend, backend_options=backend_options,
      specialize_values=["model"])
def step_eval(model, data):
    output = model.apply(data)
    return output


def cost(model, x, y, adv):
    action_pred = model.apply(x)
    log_lik = -y * np.log(action_pred)
    log_lik_adv = log_lik * adv
    loss = mean(_sum(log_lik_adv, 1))
    return loss


@myia(backend=backend, backend_options=backend_options,
      specialize_values=["model"])
def step_update(model, x, y, adv):
    adv = (adv - mean(adv)) / (std(adv) + 1e-7)
    # loss = cost(model, x, y, adv)
    loss, dmodel = value_and_grad(cost, 'model')(model, x, y, adv)
    return loss, model - dmodel


def discount_rewards(r):
    discounted_r = torch.zeros(r.size())
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add

    return discounted_r


def run_episode(net, e, env):
    state = env.reset()
    reward_sum = 0
    xs = np.array([]).astype('float32')
    ys = np.array([]).astype('float32')
    rewards = np.array([]).astype('float32')
    steps = 0

    while True:
        # env.render()

        x = np.expand_dims(state, axis=0).astype('float32')

        if xs.shape[0] == 0:
            xs = x
        else:
            xs = np.concatenate([xs, x])

        action_prob = step_eval(net, x)
        action_prob = torch.from_numpy(action_prob)

        # select an action depends on probability
        action = 0 if np.random.random() < action_prob.data[0][0] else 1

        y = np.array([[1, 0]] if action == 0 else [[0, 1]], dtype='float32')
        if ys.shape[0] == 0:
            ys = y
        else:
            ys = np.concatenate([ys, y])

        state, reward, done, _ = env.step(action)
        reward_np = np.array([[reward]], dtype='float32')
        if rewards.shape[0] == 0:
            rewards = reward_np
        else:
            rewards = np.concatenate([rewards, reward_np])
        reward_sum += reward
        steps += 1

        if done or steps >= 500:
            rewards = torch.from_numpy(rewards)
            adv = discount_rewards(rewards)
            adv = adv.numpy()
            loss, net = step_update(net, xs, ys, adv)
            HISTORY.append(reward_sum)
            print("[Episode {:>5}]  steps: {:>5} loss: {:>5} "
                  "sum(HISTORY[-5:])/5: {:>5}"
                  "".format(e, steps, loss, sum(HISTORY[-5:]) / 5))
            if sum(HISTORY[-5:]) / 5 > 490:
                return True
            else:
                return False


# for e in range(10000):
for e in range(10):
    complete = run_episode(model, e, ENV)

    if complete:
        print('complete...!')
        break
