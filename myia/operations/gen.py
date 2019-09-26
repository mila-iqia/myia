

from ..utils import core

from . import hastype


def lop(op, type, name):
    @core(name=name)
    def protocol(x, y):
        if hastype(y, type):
            return op(x, y)
        else:
            return NotImplemented
    return protocol


def rop(op, type, name):
    @core(name=name)
    def protocol(x, y):
        if hastype(y, type):
            return op(y, x)
        else:
            return NotImplemented
    return protocol


def reverse_binop(op, name):
    @core(name=name)
    def protocol(x, y):
        return op(y, x)
    return protocol
