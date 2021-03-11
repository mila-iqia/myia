from ovld import ovld

from . import data


@ovld
def to_abstract(self, xs: tuple):
    return data.AbstractStructure(
        [self(x) for x in xs], {"interface": type(xs)}
    )


@ovld
def to_abstract(self, x: object):
    return data.AbstractAtom({"interface": type(x)})


@to_abstract.variant
def precise_abstract(self, x: (int, bool)):
    return data.AbstractAtom({"value": x, "interface": type(x)})


def from_value(value, broaden=False):
    if broaden:
        return to_abstract(value)
    else:
        return precise_abstract(value)
