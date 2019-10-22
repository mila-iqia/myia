"""Operations on handles and universes."""

from ..lib import core
from . import universe_getitem, universe_setitem
from .utils import to_opdef


@to_opdef
@core(universal=True)
def handle_get(U, h):
    """Get the current value of the handle in the universe."""
    return U, universe_getitem(U, h)


@to_opdef
@core(universal=True)
def handle_set(U, h, v):
    """Set the value of the handle in the universe."""
    return universe_setitem(U, h, v), None
