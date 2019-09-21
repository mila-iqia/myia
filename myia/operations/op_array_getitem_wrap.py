"""Implementation of the 'array_getitem_wrap' operation."""

from ..lib import Slice, core, myia_static
from ..operations import array_getitem, reshape


def _dim_explicit(dim, dim_size):
    if dim < 0:
        dim = dim_size + dim
    assert dim >= 0
    return dim


@myia_static
def _build_slices(a_shp, item):
    begin = ()
    end = ()
    stride = ()
    remove_dims = ()
    for adx, a in enumerate(a_shp):
        if adx < len(item):
            i = item[adx]
            if isinstance(i, (slice, Slice)):
                begin = begin + (0 if i.start is None
                                 else _dim_explicit(i.start, a),)
                end = end + (a if i.stop is None
                             else _dim_explicit(i.stop, a),)
                stride = stride + (1 if i.step is None else i.step,)
                remove_dims = remove_dims + (False,)
            else:
                begin = begin + (_dim_explicit(i, a),)
                end = end + (_dim_explicit(i, a) + 1,)
                stride = stride + (1,)
                remove_dims = remove_dims + (True,)
        else:
            begin = begin + (0,)
            end = end + (a,)
            stride = stride + (1,)
            remove_dims = remove_dims + (False,)
    return begin, end, stride, remove_dims


@core
def array_getitem_wrap(array, item):
    """Implementation of `array_getitem`."""
    if isinstance(item, tuple):
        begin, end, stride, remove_dims = _build_slices(array.shape, item)
    else:
        begin, end, stride, remove_dims = _build_slices(array.shape, (item,))
    ret = array_getitem(array, begin, end, stride)
    final_shape = ()
    for o, r in zip(ret.shape, remove_dims):
        if not r:
            final_shape = final_shape + (o,)
    ret = reshape(ret, final_shape)
    return ret


__operation_defaults__ = {
    'name': 'array_getitem_wrap',
    'registered_name': 'array_getitem_wrap',
    'mapping': array_getitem_wrap,
    'python_implementation': None,
}
