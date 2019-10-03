
from . import universe_getitem, universe_setitem
from ..lib import core
from .utils import to_opdef


@to_opdef
@core(universal=True)
def handle_get(U, h):
    return U, universe_getitem(U, h)


@to_opdef
@core(universal=True)
def handle_set(U, h, v):
    return universe_setitem(U, h, v), None
