"""Tools for aliasing detection and handling."""

from . import data as ab
from ..prim import ops as P
from ..utils import overload


@overload(bootstrap=True)
def generate_getters(self, tup: ab.AbstractTuple, get):
    """Recursively generate sexps for getting elements of a data structure."""
    yield tup, get
    for i, elem in enumerate(tup.elements):
        geti = (P.tuple_getitem, get, i)
        yield from self(elem, geti)


@overload  # noqa: F811
def generate_getters(self, dat: ab.AbstractClassBase, get):
    yield dat, get
    for k, elem in dat.attributes.items():
        getk = (P.record_getitem, get, k)
        yield from self(elem, getk)


@overload  # noqa: F811
def generate_getters(self, dat: ab.AbstractDict, get):
    yield dat, get
    for k, elem in dat.entries.items():
        getk = (P.dict_getitem, get, k)
        yield from self(elem, getk)


@overload  # noqa: F811
def generate_getters(self, obj: object, get):
    yield obj, get


def setter_from_getter(getter, value):
    """Generate an expression to set a value from the expression to get it."""
    setters = {
        P.tuple_getitem: P.tuple_setitem,
        P.dict_getitem: P.dict_setitem,
        P.record_getitem: P.record_setitem,
    }
    if isinstance(getter, tuple):
        oper, elem, i = getter
        return setter_from_getter(elem, (setters[oper], elem, i, value))
    else:
        return value
