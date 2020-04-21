"""Operations on handles and universes."""

from .. import xtype
from ..lib import ANYTHING, TYPE, VALUE, AbstractScalar, HandleInstance, core
from . import make_handle, typeof, universe_getitem, universe_setitem
from .utils import to_opdef


@to_opdef
@core(universal=True)
def make_cell(init, U):
    """Create a new cell."""
    U, h = make_handle(typeof(init), U)
    return universe_setitem(U, h, init), h


@to_opdef
@core(universal=True)
def cell_get(h, U):
    """Get the current value of the cell."""
    return U, universe_getitem(U, h)


@to_opdef
@core(universal=True)
def cell_set(h, v, U):
    """Set the value of the cell."""
    return universe_setitem(U, h, v), None
