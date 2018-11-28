
import math
import numpy
from dataclasses import is_dataclass
from functools import reduce

from .. import dtype, dshape
from ..debug.utils import mixin
from ..infer import ANYTHING, InferenceError, MyiaTypeError
from ..infer.core import Pending, Later
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


class PartialApplication:
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args

    def __hash__(self):
        return hash((self.fn, self.args))

    def __eq__(self, other):
        return isinstance(other, PartialApplication) \
            and self.fn == other.fn \
            and self.args == other.args


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
    def __init__(self, values, count=10):
        self.values = values
        self.count = count

    def build(self, name):
        v = self.values.get(name, ABSENT)
        if v is not ABSENT:
            if isinstance(v, Pending):
                return v.result()
            else:
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

    def broaden(self):
        return self

    def merge(self, other):
        if type(self) is not type(other):
            raise MyiaTypeError(f'Expected {type(self).__name__}')
        rval = self.merge_structure(other)
        for track in ['type', 'value', 'shape']:
            v1 = self.values.get(track, ABSENT)
            v2 = other.values.get(track, ABSENT)
            method = getattr(self, f'merge_{track}')
            rval.values[track] = method(v1, v2)
        return rval

    def merge_value(self, v1, v2):
        if v1 == v2:
            return v1
        elif isinstance(v1, Possibilities):
            return Possibilities(v1 | v2)
        else:
            return ANYTHING

    def merge_type(self, v1, v2):
        if v1 != v2:
            raise MyiaTypeError(f'Cannot merge {v1} and {v2} (3)')
        return v1

    def merge_shape(self, v1, v2):
        if v1 != v2:
            raise MyiaTypeError(f'Cannot merge {v1} and {v2} (shp)')
        return v1

    def accept(self, other):
        raise NotImplementedError()

    def make_key(self):
        return tuple(sorted(self.values.items()))

    def _resolve(self, key, value):
        self.values[key] = value

    def __repr__(self):
        contents = [f'{k}={v}' for k, v in self.values.items()]
        return f'V({", ".join(contents)})'


class AbstractScalar(AbstractValue):

    def merge_structure(self, other):
        return AbstractScalar({})

    def __repr__(self):
        contents = [f'{k}={v}' for k, v in self.values.items()]
        return f'S({", ".join(contents)})'


class AbstractTuple(AbstractValue):
    def __init__(self, elements, values=None):
        super().__init__(values or {})
        self.elements = tuple(elements)

    def merge_structure(self, other):
        assert len(self.elements) == len(other.elements)
        return AbstractTuple(
            [x.merge(y) for x, y in zip(self.elements, other.elements)]
        )

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

    def merge_structure(self, other):
        return AbstractArray(self.element.merge(other.element))

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
# Broaden #
###########


def _broaden_values(d):
    return {
        'value': broaden(d['value']),
        'type': d['type'],
        'shape': d['shape'],
    }

@overload(bootstrap=True)
def broaden(self, x: AbstractScalar):
    return AbstractScalar(_broaden_values(x.values))


@overload(bootstrap=True)
def broaden(self, x: AbstractTuple):
    return AbstractTuple(
        [self(y) for y in x.elements],
        _broaden_values(x.values)
    )


@overload(bootstrap=True)
def broaden(self, x: object):
    return ANYTHING


#########
# Merge #
#########


@overload(bootstrap=True)
def _amerge(self, x1: Possibilities, x2, loop, forced):
    if x1.issuperset(x2):
        return x1
    if forced:
        raise MyiaTypeError('Cannot merge Possibilities')
    else:
        return Possibilities(x1 | x2)


@overload
def _amerge(self, x1: dtype.TypeMeta, x2, loop, forced):
    if x1 != x2:
        raise MyiaTypeError(f'Cannot merge {x1} and {x2}')
    return x1


@overload
def _amerge(self, x1: dict, x2, loop, forced):
    if set(x1.keys()) != set(x2.keys()):
        raise MyiaTypeError(f'Keys mismatch')
    changes = False
    rval = {}
    for k, v in x1.items():
        res = amerge(v, x2[k], loop, forced)
        if res is not v:
            changes = True
        rval[k] = res
    return x1 if forced or not changes else rval


@overload
def _amerge(self, x1: tuple, x2, loop, forced):
    if len(x1) != len(x2):
        raise MyiaTypeError(f'Tuple length mismatch')
    changes = False
    rval = []
    for v1, v2 in zip(x1, x2):
        res = amerge(v1, v2, loop, forced)
        if res is not v1:
            changes = True
        rval.append(res)
    return x1 if forced or not changes else tuple(rval)


@overload
def _amerge(self, x1: AbstractScalar, x2, loop, forced):
    values = amerge(x1.values, x2.values, loop, forced)
    if forced or values is x1.values:
        return x1
    return AbstractScalar(values)


@overload
def _amerge(self, x1: AbstractTuple, x2, loop, forced):
    args1 = (x1.elements, x1.values)
    args2 = (x2.elements, x2.values)
    merged = amerge(args1, args2, loop, forced)
    if forced or merged is args1:
        return x1
    return AbstractTuple(*merged)


@overload
def _amerge(self, x1: AbstractArray, x2, loop, forced):
    args1 = (x1.element, x1.values)
    args2 = (x2.element, x2.values)
    merged = amerge(args1, args2, loop, forced)
    if forced or merged is args1:
        return x1
    return AbstractArray(*merged)


@overload
def _amerge(self, x1: AbstractList, x2, loop, forced):
    args1 = (x1.element, x1.values)
    args2 = (x2.element, x2.values)
    merged = amerge(args1, args2, loop, forced)
    if forced or merged is args1:
        return x1
    return AbstractList(*merged)


@overload
def _amerge(self, x1: (int, float, bool), x2, loop, forced):
    if x1 is ANYTHING:
        return x1
    if forced:
        if x1 != x2:
            raise MyiaTypeError(f'Cannot merge {x1} and {x2}')
    elif x2 is ANYTHING or x1 != x2:
        return ANYTHING
    return x1


@overload
def _amerge(self, x1: object, x2, loop, forced):
    if x1 is ANYTHING:
        if x2 is ANYTHING:
            return x1
        return self[type(x2)](x1, x2, loop, forced)
    if x1 != x2:
        raise MyiaTypeError(f'Cannot merge {x1} and {x2}')
    return x1


def amerge(x1, x2, loop, forced, accept_pending=True):
    isp1 = isinstance(x1, Pending)
    isp2 = isinstance(x2, Pending)
    if (isp1 or isp2) and not accept_pending:
        raise AssertionError('Cannot have Pending here.')
    if isp1 and isp2:
        return bind(loop, x1 if forced else None, [], [x1, x2])
    elif isp1:
        return bind(loop, x1 if forced else None, [x2], [x1])
    elif isp2:
        return bind(loop, x1 if forced else None, [x1], [x2])
    elif x1 is ANYTHING:
        if x2 is ANYTHING:
            return x1
        return _amerge[type(x2)](x1, x2, loop, forced)
    elif x2 is not ANYTHING and type(x1) is not type(x2):
        raise MyiaTypeError(
            f'Type mismatch: {type(x1)} != {type(x2)}; {x1} != {x2}'
        )
    else:
        return _amerge(x1, x2, loop, forced)


def is_simple(x):
    if dtype.ismyiatype(x):
        return True
    else:
        return False


def bind(loop, committed, resolved, pending):

    def amergeall():
        if committed is None:
            v = reduce(lambda x1, x2: amerge(x1, x2,
                                             loop=loop,
                                             forced=False,
                                             accept_pending=False),
                       resolved)
        else:
            v = reduce(lambda x1, x2: amerge(x1, x2,
                                             loop=loop,
                                             forced=True,
                                             accept_pending=False),
                       resolved, committed)
        return v

    resolved = list(resolved)
    pending = set(pending)
    assert pending

    def resolve(fut):
        nonlocal committed
        pending.remove(fut)
        result = fut.result()
        if fut is committed:
            committed = result
        resolved.append(result)
        if not pending:
            v = amergeall()
            if rval is not None and not rval.done():
                rval.resolve_to(v)

    for p in pending:
        p.add_done_callback(resolve)

    def premature_resolve():
        nonlocal committed
        if not resolved and committed is None:
            raise Later()
        committed = amergeall()
        committed = broaden(committed)
        resolved.clear()
        return committed

    if resolved and all(is_simple(x) for x in resolved):
        rval = None
        for p in pending:
            p.equiv.update(resolved)
            p.equiv.update(resolved)
        return resolved[0]
    else:
        priority = (1000
                    if any(p.priority is None for p in pending)
                    else min(p.priority for p in pending) - 1)
        rval = loop.create_pending(
            resolve=premature_resolve,
            priority=priority,
        )
        rval.equiv.update(resolved)
        rval.equiv.update(pending)
        for p in pending:
            p.tie(rval)

        return rval



    # def abstract_merge(self, *values):
    #     resolved = []
    #     pending = set()
    #     committed = None
    #     for v in values:
    #         if isinstance(v, Pending):
    #             if v.resolved():
    #                 resolved.append(v.result())
    #             else:
    #                 pending.add(v)
    #         else:
    #             resolved.append(v)

    #     if pending:
    #         def resolve(fut):
    #             pending.remove(fut)
    #             result = fut.result()
    #             resolved.append(result)
    #             if not pending:
    #                 v = self.force_merge(resolved, model=committed)
    #                 rval.resolve_to(v)

    #         for p in pending:
    #             p.add_done_callback(resolve)

    #         def premature_resolve():
    #             nonlocal committed
    #             committed = self.force_merge(resolved)
    #             resolved.clear()
    #             return committed

    #         rval = self.engine.loop.create_pending(
    #             resolve=premature_resolve,
    #             priority=-1,
    #         )
    #         rval.equiv.update(values)
    #         for p in pending:
    #             p.tie(rval)
    #         return rval
    #     else:
    #         return self.force_merge(resolved)



###########
# Broaden #
###########


@overload(bootstrap=True)
def abroaden(x: object, count):
    if x is ANYTHING or count == 0:
        return ANYTHING
    else:
        return x


@overload
def abroaden(x: dict, count):
    pass


###########
# Cleanup #
###########


# @overload(bootstrap=True)
# def reify_sync(self, p: Pending):
#     return p.result()


# @overload
# def reify_sync(self, d: dict):
#     rval = {}
#     changes = False
#     for k, v in d.items():
#         v2 = self(v)
#         if v2 is not v:
#             changes = True
#         rval[k] = v2
#     return rval if changes else d


# @overload
# def reify_sync(self, tup: tuple):
#     rval = []
#     changes = False
#     for x in tup:
#         x2 = self(x)
#         if x2 is not x:
#             changes = True
#         rval.append(x2)
#     return rval if changes else tup


# @overload
# def reify_sync(self, v: AbstractValue):
#     d2 = reify_sync(v.values)
#     if d2 is v.values:
#         return v
#     return AbstractValue(d2)


# @overload
# def reify_sync(self, v: AbstractValue):
#     d2 = reify_sync(v.values)
#     if d2 is v.values:
#         return v
#     return AbstractValue(d2)


# @overload
# def reify_sync(self, v: object):
#     return v


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
