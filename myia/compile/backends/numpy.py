import numpy as np

from . import Backend
from ... import dtype


class NumPyBackend(Backend):
    """Backend that uses the debug VM and numpy."""

    def from_numpy(self, n):
        """Returns n."""
        return n

    def to_numpy(self, n):
        """Returns n."""
        return n

    def from_scalar(self, s, dt):
        """Returns s."""
        return s

    def to_scalar(self, s):
        """Returns s."""
        return s

    def check_array(self, arg, t):
        """Checks that arg has elements of the right dtype."""
        if not isinstance(arg, np.ndarray):
            raise TypeError('Expected ndarray')
        if arg.dtype != dtype.type_to_np_dtype(t):
            raise TypeError('Wrong dtype')
        return arg

    def configure(self, pip):
        from myia.pipeline.steps import step_debug_export
        return pip.insert_after('compile', export=step_debug_export) \
                  .configure(compile=False, backend=self)
