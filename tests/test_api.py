import pytest

from myia.api import checked


@checked
def fact(x):
    if x <= 1:
        return x
    else:
        return x * fact(x - 1)


class Banana:
    def __init__(self, size):
        self.size = size

    @checked
    def bigger(self, obj):
        return obj <= self.size


def test_good():
    assert fact(5) == 120

    # Should have been cached
    assert fact(5) == 120


def test_method():
    b = Banana(7)
    assert b.bigger(4) is True
