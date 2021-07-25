"""Inferrers for basic functions and the standard library."""

import operator
import types

from .. import basics
from ..abstract import data
from ..abstract.map import MapError
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
    _ = yield Require(cond)  # TODO: check bool
    ift_t = yield Require(ift)
    iff_t = yield Require(iff)
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
    obj_node, key_node = args
    assert key_node.is_constant(int)
    obj = yield Require(obj_node)
    key = key_node.value
    if not (isinstance(obj, data.AbstractStructure) and obj.tracks.interface in (tuple, list)):
        raise AssertionError(f"getitem can currently only be used for lists and tuples, got {obj}[{key}]")
    return obj.elements[key]


def make_tuple_inferrer(node, args, unif):
    tuple_types = []
    for arg_node in args:
        tuple_types.append((yield Require(arg_node)))
    return data.AbstractStructure(tuple_types, {"interface": tuple})


def make_list_inferrer(node, args, unif):
    tuple_types = []
    for arg_node in args:
        tuple_types.append((yield Require(arg_node)))
    for typ in tuple_types[1:]:
        if typ is not tuple_types[0]:
            raise MapError(tuple_types[0], typ, "list elements don't have same type")
    return data.AbstractStructure([tuple_types[0]] if tuple_types else [], {"interface": list})


X = data.Generic("x")

AbstractNone = data.AbstractAtom({"interface": type(None)})

def add_standard_inferrers(inferrers):
    """Register all the inferrers in this file."""
    inferrers.update(
        {
            operator.add: signature(X, X, ret=X),
            operator.and_: signature(X, X, ret=X),
            operator.eq: signature(X, X, ret=bool),
            operator.gt: signature(X, X, ret=bool),
            operator.invert: signature(X, ret=X),
            operator.le: signature(X, X, ret=bool),
            operator.lshift: signature(X, X, ret=X),
            operator.mul: signature(X, X, ret=X),
            operator.neg: signature(X, ret=X),
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
            basics.global_universe_setitem: signature(
                data.AbstractStructure([X], tracks={"interface": Handle}),
                X,
                ret=AbstractNone
            ),
            basics.partial: inference_function(partial_inferrer),
            basics.make_tuple: inference_function(make_tuple_inferrer),
            basics.make_list: inference_function(make_list_inferrer),
        }
    )
