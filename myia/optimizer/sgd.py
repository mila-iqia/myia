import numpy as np
from dataclasses import dataclass


# SGD function
@dataclass()
class SGD():
    lr: float = 0.1
    momentum: float = 0
    decay: float = 0.001
    nesterov: bool = False

    # init
    def __post_init__(self):
        if self.lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(self.lr))
        if self.momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(self.momentum))
        if self.decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(self.decay))

        defaults = dict(lr=self.lr, momentum=self.momentum, weight_decay=self.decay)
        self.defaults = defaults
        self.iterations = 0
        self.lr_current_iter = self.lr


    # run before apply
    def update(self):
        if self.decay > 0:
            self.lr_current_iter = self.lr * (1. / (1. + self.decay * self.iterations))
        self.iterations += 1 

    # update
    def step(self, param, grad, moment):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # print(self)
        # print(self.iterations)
        if (self.iterations == 0):
            moment = np.zeros_like(param)

        lr = self.lr_current_iter
        v = self.momentum * moment - lr * grad

        if self.nesterov:
            new_p = param + self.momentum * v - lr * grad
        else:
            new_p = param + v

        moment = v
        param = new_p

        return param, moment