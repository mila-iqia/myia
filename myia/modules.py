"""Consolidate all Myia dataclass modules in a single module."""

from dataclasses import dataclass

import myia.operations as ops
import myia.public_api as pub
from myia import ArithmeticData


@dataclass(frozen=True)
class Linear(ArithmeticData):
    """Linear layer."""

    W: "Weights array"
    b: "Biases vector"

    def apply(self, input):
        """Apply the layer."""
        return input @ self.W + self.b


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
class Softmax(ArithmeticData):
    """LogSoftmax layer."""

    # dims: Static(int)
    dims: "Dimensions tuple"

    def apply(self, input):
        """Apply the layer."""
        return pub.softmax(input, dim=self.dims)


@dataclass(frozen=True)
class Tanh(ArithmeticData):
    """Tanh layer."""

    def apply(self, input):
        """Apply the layer."""
        return ops.array_tanh(input)


__all__ = ["Linear", "Sequential", "Softmax", "Tanh"]
