"""Inferrers for basic functions and the standard library."""

import operator
import types

from .. import basics
from ..abstract import data
from ..abstract.to_abstract import precise_abstract
from ..basics import Handle
from ..ir import Constant
from ..utils.misc import ModuleNamespace
from .algo import Require, RequireAll
from .infnode import Replace, inference_function, signature


def resolve(node, args, unif):
    """Inferrer for the resolve basic function."""
    ns, name = args
    assert ns.is_constant(ModuleNamespace)
    assert name.is_constant(str)
    resolved = ns.value[name.value]
    if hasattr(resolved, "__myia__"):
        resolved = resolved.__myia__()
    ct = Constant(resolved)
    yield Replace(ct)
    res = yield Require(ct)
    return res


def user_switch(node, args, unif):
    """Inferrer for the user_switch basic function."""
    cond, ift, iff = args
    cond_type = yield Require(cond)  # TODO: check bool
    ift_t = yield Require(ift)
    iff_t = yield Require(iff)
    # If both branches return same type, then return this type.
    if ift_t is iff_t:
        return ift_t
    # If cond_type has a value, check it.
    if cond_type is precise_abstract(True):
        return ift_t
    if cond_type is precise_abstract(False):
        return iff_t
    # Otherwise, return an union of both types.
    return data.AbstractUnion([ift_t, iff_t], tracks={})


def partial_inferrer(node, args, unif):
    """Inferrer for the a partial application."""
    fn, *args = args
    fn_type = yield Require(fn)
    arg_types = yield RequireAll(*args)
    return data.AbstractStructure(
        (*fn_type.elements, *arg_types),
        tracks={"interface": fn_type.tracks.interface},
    )


def getattr_inferrer(node, args, unif):
    """Inferrer for the getattr function."""
    obj_node, key_node = args
    assert key_node.is_constant(str)
    obj = yield Require(obj_node)
    key = key_node.value
    interface = obj.tracks.interface
    result = getattr(interface, key)
    if isinstance(result, (types.MethodType, types.WrapperDescriptorType)):
        ct = Constant(result)
        new_node = node.graph.apply(basics.partial, ct, obj_node)
    elif isinstance(interface, types.ModuleType) and callable(result):
        new_node = Constant(result)
    else:
        raise AssertionError("getattr can currently only be used for methods")
        # new_node = Constant(result)
    yield Replace(new_node)
    res = yield Require(new_node)
    return res


def getitem_inferrer(node, args, unif):
    """Inferrer for the getitem function."""
    obj_node, key_node = args
    obj = yield Require(obj_node)
    if isinstance(obj, data.AbstractDict):
        key = yield Require(key_node)
        key_pos = obj.keys.index(key)
        return obj.values[key_pos]
    else:
        assert key_node.is_constant(int), key_node
        key = key_node.value
        if not (
            isinstance(obj, data.AbstractStructure)
            and obj.tracks.interface in (tuple, list)
        ):
            raise AssertionError(
                f"getitem can currently only be used for "
                f"dicts, lists and tuples, got {obj}[{key}]"
            )
        return obj.elements[key]


def make_tuple_inferrer(node, args, unif):
    """Inferrer for the make_tuple function."""
    tuple_types = []
    for arg_node in args:
        tuple_types.append((yield Require(arg_node)))
    return data.AbstractStructure(tuple_types, {"interface": tuple})


def make_list_inferrer(node, args, unif):
    """Inferrer for the make_list function."""
    tuple_types = []
    for arg_node in args:
        tuple_types.append((yield Require(arg_node)))
    elements = []
    if tuple_types:
        from myia.abstract import utils as autils

        base_type = tuple_types[0]
        for typ in tuple_types[1:]:
            base_type = autils.merge(base_type, typ, U=unif)[0]
        elements.append(base_type)
    return data.AbstractStructure(elements, {"interface": list})


def make_dict_inferrer(node, args, unif):
    """Inferrer for the make_dict function."""
    assert (
        not len(args) % 2
    ), f"make_dict: expected even number of arguments, got {len(args)}"
    arg_types = yield RequireAll(*args)
    return data.AbstractDict(arg_types)


def len_inferrer(node, args, unif):
    obj_node, = args
    obj_type = yield Require(obj_node)
    interface = obj_type.tracks.interface
    if not hasattr(interface, "__len__"):
        raise AttributeError(f"Interface has no attribute __len__: {interface}")
    return data.AbstractAtom({"interface": int})


def myia_iter_inferrer(node, args, unif):
    """Inferrer for the myia_iter function."""
    (iterable_node,) = args
    iterable_type = yield Require(iterable_node)
    if iterable_type.tracks.interface in (tuple, list, range):
        return iterable_type
    raise TypeError(f"myia_iter: unexpected input type: {iterable_type}")


def myia_hasnext_inferrer(node, args, unif):
    """Inferrer for the myia_hasnext function."""
    (iterable_node,) = args
    iterable_type = yield Require(iterable_node)
    if iterable_type.tracks.interface in (range, list):
        return data.AbstractAtom({"interface": bool})
    if iterable_type.tracks.interface is tuple:
        assert isinstance(iterable_type, data.AbstractStructure)
        return precise_abstract(bool(iterable_type.elements))
    raise TypeError(f"myia_hasnext: unexpected input type: {iterable_type}")


def myia_next_inferrer(node, args, unif):
    """Inferrer for the myia_next function."""
    (iterable_node,) = args
    iterable_type = yield Require(iterable_node)
    if iterable_type.tracks.interface is range:
        return data.AbstractStructure(
            [
                data.AbstractAtom({"interface": int}),
                data.AbstractAtom({"interface": range}),
            ],
            {"interface": tuple},
        )
    if iterable_type.tracks.interface is tuple:
        assert isinstance(iterable_type, data.AbstractStructure)
        type_next_el = iterable_type.elements[0]
        type_next_seq = iterable_type.elements[1:]
        return data.AbstractStructure(
            [
                type_next_el,
                data.AbstractStructure(type_next_seq, {"interface": tuple}),
            ],
            {"interface": tuple},
        )
    if iterable_type.tracks.interface is list:
        assert isinstance(iterable_type, data.AbstractStructure)
        (type_el,) = iterable_type.elements
        return data.AbstractStructure(
            [
                type_el,
                data.AbstractStructure(
                    [type_el], {"interface": iterable_type.tracks.interface}
                ),
            ],
            {"interface": tuple},
        )
    raise TypeError(f"myia_next: unexpected input type: {iterable_type}")


X = data.Generic("x")
Y = data.Generic("y")


def add_standard_inferrers(inferrers):
    """Register all the inferrers in this file."""
    inferrers.update(
        {
            len: inference_function(len_inferrer),
            hasattr: signature(X, str, ret=bool),
            operator.add: signature(X, X, ret=X),
            operator.and_: signature(X, X, ret=X),
            operator.eq: signature(X, X, ret=bool),
            operator.gt: signature(X, Y, ret=bool),
            operator.invert: signature(X, ret=X),
            operator.is_: signature(X, Y, ret=bool),
            operator.is_not: signature(X, Y, ret=bool),
            operator.le: signature(X, X, ret=bool),
            operator.lshift: signature(X, X, ret=X),
            operator.lt: signature(X, X, ret=bool),
            operator.mul: signature(X, X, ret=X),
            operator.neg: signature(X, ret=X),
            operator.not_: signature(X, ret=bool),
            operator.or_: signature(X, X, ret=X),
            operator.rshift: signature(X, X, ret=X),
            operator.sub: signature(X, X, ret=X),
            operator.truth: signature(X, ret=bool),
            operator.xor: signature(X, X, ret=X),
            basics.return_: signature(X, ret=X),
            basics.resolve: inference_function(resolve),
            basics.user_switch: inference_function(user_switch),
            int.__add__: signature(int, int, ret=int),
            float.__add__: signature(float, float, ret=float),
            getattr: inference_function(getattr_inferrer),
            operator.getitem: inference_function(getitem_inferrer),
            type: signature(
                X,
                ret=data.AbstractStructure([X], tracks={"interface": type}),
            ),
            basics.make_handle: signature(
                data.AbstractStructure([X], tracks={"interface": type}),
                ret=data.AbstractStructure([X], tracks={"interface": Handle}),
            ),
            basics.global_universe_getitem: signature(
                data.AbstractStructure([X], tracks={"interface": Handle}),
                ret=X,
            ),
            basics.partial: inference_function(partial_inferrer),
            basics.global_universe_setitem: signature(
                data.AbstractStructure([X], tracks={"interface": Handle}),
                X,
                ret=None,
            ),
            basics.make_tuple: inference_function(make_tuple_inferrer),
            basics.make_list: inference_function(make_list_inferrer),
            basics.myia_iter: inference_function(myia_iter_inferrer),
            basics.myia_hasnext: inference_function(myia_hasnext_inferrer),
            basics.myia_next: inference_function(myia_next_inferrer),
            basics.make_dict: inference_function(make_dict_inferrer),
        }
    )
