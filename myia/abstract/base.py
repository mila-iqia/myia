
import numpy
from dataclasses import is_dataclass

from .. import dtype, dshape
from ..debug.utils import mixin
from ..infer import ANYTHING, InferenceError
from ..utils import overload, UNKNOWN, Named


ABSENT = Named('ABSENT')


#################
# Abstract data #
#################


class Possibilities(frozenset):
    pass


class GraphAndContext:
    def __init__(self, graph, context):
        self.graph = graph
        self.context = context

    def __hash__(self):
        return hash((self.graph, self.context))

    def __eq__(self, other):
        return isinstance(other, GraphAndContext) \
            and self.graph == other.graph \
            and self.context == other.context


class AbstractBase:

    def make_key(self):
        raise NotImplementedError()

    def key(self):
        if not hasattr(self, '_key'):
            self._key = self.make_key()
        return self._key

    def __eq__(self, other):
        return type(self) is type(other) \
            and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())


class AbstractValue(AbstractBase):
    def __init__(self, values):
        self.values = values

    def build(self, name):
        v = self.values.get(name, ABSENT)
        if v is not ABSENT:
            return v
        else:
            method = getattr(self, f'build_{name}')
            return method()

    def build_value(self):
        raise NotImplementedError()

    def build_type(self):
        raise NotImplementedError()

    def build_shape(self):
        raise NotImplementedError()

    def make_key(self):
        return tuple(sorted(self.values.items()))

    def _resolve(self, key, value):
        self.values[key] = value

    def __repr__(self):
        contents = [f'{k}={v}' for k, v in self.values.items()]
        return f'V({", ".join(contents)})'


class AbstractScalar(AbstractValue):
    def __repr__(self):
        contents = [f'{k}={v}' for k, v in self.values.items()]
        return f'S({", ".join(contents)})'


class AbstractTuple(AbstractValue):
    def __init__(self, elements):
        super().__init__({})
        self.elements = tuple(elements)

    def build_value(self):
        return tuple(e.build('value') for e in self.elements)

    def build_type(self):
        return dtype.Tuple[[e.build('type') for e in self.elements]]

    def build_shape(self):
        return dshape.TupleShape([e.build('shape') for e in self.elements])

    def make_key(self):
        return (super().make_key(), self.elements)

    def __repr__(self):
        return f'T({", ".join(map(repr, self.elements))})'


class AbstractArray(AbstractValue):
    def __init__(self, element, values=None):
        super().__init__(values or {})
        self.element = element

    def build_type(self):
        return dtype.Array[self.element.build('type')]

    def make_key(self):
        return (super().make_key(), self.element)

    def __repr__(self):
        return f'A({self.element}, shape={self.values["shape"]})'


class AbstractList(AbstractValue):
    def __init__(self, element, values=None):
        super().__init__(values or {})
        self.element = element

    def build_type(self):
        return dtype.List[self.element.build('type')]

    def build_shape(self):
        return dshape.ListShape(self.element.build('shape'))

    def make_key(self):
        return (super().make_key(), self.element)

    def __repr__(self):
        return f'L({self.element})'


class AbstractClass(AbstractValue):
    def __init__(self, tag, attributes, methods):
        self.tag = tag
        self.attributes = attributes
        self.methods = methods

    def build_type(self):
        return dtype.Class[
            self.tag,
            {name: x.build('type')
             for name, x in self.attributes.items()},
            self.methods
        ]

    def build_shape(self):
        return dshape.ClassShape(
            {name: x.build('shape')
             for name, x in self.attributes.items()},
        )

    def make_key(self):
        return (tag, tuple(sorted(self.attributes.items())))

    def __repr__(self):
        elems = [f'{k}={v}' for k, v in self.attributes.items()]
        return f'{self.tag}({", ".join(elems)})'


# class AbstractFunction:
#     pass


# class AbstractMonoFunction:
#     pass


# class AbstractPolyFunction:
#     pass


# class AbstractPartial:
#     pass


# class Merged:
#     pass


#############
# From vref #
#############


@overload(bootstrap=True)
def from_vref(self, v, t: dtype.Tuple, s):
    elems = []
    for i, tt in enumerate(t.elements):
        vv = v[i] if isinstance(v, tuple) else ANYTHING
        ss = s.shape[i]
        elems.append(self(vv, tt, ss))
    return AbstractTuple(elems)


@overload
def from_vref(self, v, t: dtype.Array, s):
    vv = ANYTHING
    tt = t.elements
    ss = dshape.NOSHAPE
    return AbstractArray(self(vv, tt, ss), {'shape': s})


@overload
def from_vref(self, v, t: dtype.List, s):
    vv = ANYTHING
    tt = t.element_type
    ss = s.shape
    return AbstractList(self(vv, tt, ss), {})


@overload
def from_vref(self, v, t: (dtype.Number, dtype.Bool, dtype.External), s):
    return AbstractScalar({'value': v, 'type': t, 'shape': s})


@overload
def from_vref(self, v, t: dtype.Class, s):
    attrs = {}
    for k, tt in t.attributes.items():
        vv = ANYTHING if v in (ANYTHING, UNKNOWN) else getattr(v, k)
        ss = ANYTHING if s in (ANYTHING, UNKNOWN) else s.shape[k]
        attrs[k] = self(vv, tt, ss)
    return AbstractClass(
        t.tag,
        attrs,
        t.methods
        # {'value': v, 'type': t, 'shape': s}
    )


@overload
def from_vref(self, v, t: dtype.TypeMeta, s):
    return self[t](v, t, s)


###########
# shapeof #
###########


def shapeof(v):
    """Infer the shape of a constant."""
    if isinstance(v, tuple):
        return dshape.TupleShape(shapeof(e) for e in v)
    elif isinstance(v, list):
        shps = [shapeof(e) for e in v]
        if len(shps) == 0:  # pragma: no cover
            # from_value of the type track will fail before this
            raise InferenceError('Cannot infer the shape of []')
        return dshape.ListShape(dshape.find_matching_shape(shps))
    elif is_dataclass(v):
        if isinstance(v, type):
            assert False
            # rec = self.constructors[P.make_record](self)
            # typ = pytype_to_myiatype(v)
            # vref = self.engine.vref({'value': typ, 'type': TypeType})
            # return PartialInferrer(self, rec, [vref])
        else:
            return dshape.ClassShape(
                dict((n, shapeof(getattr(v, n)))
                     for n in v.__dataclass_fields__.keys()))
    elif isinstance(v, numpy.ndarray):
        return v.shape
    else:
        return dshape.NOSHAPE


##################
# Representation #
##################


def _clean(values):
    return {k: v for k, v in values.items()
            if v not in {ANYTHING, UNKNOWN, dshape.NOSHAPE}}


@mixin(AbstractValue)
class _AbstractValue:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object(
            '★Value',
            _clean(self.values).items(),
            delimiter="↦",
            cls='abstract',
        )


@mixin(AbstractTuple)
class _AbstractTuple:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_iterable(
            self.elements,
            before='★T',
            cls='abstract',
        )


@mixin(AbstractArray)
class _AbstractArray:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_iterable(
            [
                H.div(
                    hrepr.stdrepr_object(
                        '', _clean(self.values).items(), delimiter="↦",
                        cls='noborder'
                    ),
                    hrepr(self.element),
                    style='display:flex;flex-direction:column;'
                )
            ],
            before='★A',
            cls='abstract',
        )


@mixin(AbstractList)
class _AbstractList:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_iterable(
            [
                H.div(
                    hrepr.stdrepr_object(
                        '', _clean(self.values).items(), delimiter="↦",
                        cls='noborder'
                    ),
                    hrepr(self.element),
                    style='display:flex;flex-direction:column;'
                )
            ],
            before='★L',
            cls='abstract',
        )


@mixin(AbstractClass)
class _AbstractClass:
    def __hrepr__(self, H, hrepr):
        return hrepr.stdrepr_object(
            f'★{self.tag}',
            self.attributes.items(),
            delimiter="↦",
            cls='abstract'
        )
