"""Operations on handles and universes."""

from .. import xtype
from ..lib import ANYTHING, TYPE, VALUE, AbstractScalar, HandleInstance, core
from . import make_handle, universe_getitem, universe_setitem
from .utils import to_opdef

cell_id = HandleInstance(
    0,
    abstract=AbstractScalar({VALUE: ANYTHING, TYPE: xtype.Int[64]}),
    id="cell_id",
)


# @to_opdef
# @core(universal=True)
# def make_cell(init, U):
#     """Create a new cell."""
#     curr_cell_id = universe_getitem(U, cell_id)
#     next_cell_id = curr_cell_id + 1
#     h = make_handle(next_cell_id, init)
#     return universe_setitem(U, cell_id, next_cell_id), h


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
