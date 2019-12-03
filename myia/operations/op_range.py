"""Implementation of the 'range' operation."""

from dataclasses import dataclass

from ..lib import core


@dataclass
class Range:  # pragma: no cover
    """Implement a Range in Myia."""

    start: object
    stop: object
    step: object

    @core
    def __myia_iter__(self):
        return self

    @core
    def __myia_next__(self):
        return self.start, Range(self.start + self.step, self.stop, self.step)

    @core
    def __myia_hasnext__(self):
        return self.start < self.stop


@core
def range_(start, stop=None, step=None):
    """Myia implementation of the standard range function."""
    if stop is None:
        stop = start
        start = 0
    if step is None:
        step = 1
    return Range(start, stop, step)


__operation_defaults__ = {
    'name': 'range',
    'registered_name': 'range',
    'mapping': range_,
    'python_implementation': range,
}
