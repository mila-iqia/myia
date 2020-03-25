"""Definitions for the primitive `array_scan`."""

import numpy as np

from . import primitives as P


def pyimpl_array_scan(fn, init, array, axis):
    """Implement `array_scan`."""
    # This is inclusive scan because it's easier to implement
    # We will have to discuss what semantics we want later
    def f(ary):
        val = init
        it = np.nditer([ary, None])
        for x, y in it:
            val = fn(val, x)
            y[...] = val
        return it.operands[1]

    return np.apply_along_axis(f, axis, array)


def debugvm_array_scan(vm, fn, init, array, axis):
    """Implement `array_scan` for the debug VM."""

    def fn_(a, b):
        return vm.call(fn, [a, b])

    return pyimpl_array_scan(fn_, init, array, axis)


__operation_defaults__ = {
    "name": "array_scan",
    "registered_name": "array_scan",
    "mapping": P.array_scan,
    "python_implementation": pyimpl_array_scan,
}


__primitive_defaults__ = {
    "name": "array_scan",
    "registered_name": "array_scan",
    "type": "backend",
    "python_implementation": pyimpl_array_scan,
    "debugvm_implementation": debugvm_array_scan,
    "inferrer_constructor": None,
    "grad_transform": None,
}
