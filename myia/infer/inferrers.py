"""Inferrers for basic functions and the standard library."""

import operator

from .. import basics
from ..abstract import data
from ..basics import Handle
from ..ir import Constant
from ..utils.misc import ModuleNamespace
from .algo import Require
from .infnode import Replace, inference_function, signature


def resolve(node, unif):
    """Inferrer for the resolve basic function."""
    ns, name = node.inputs
    assert ns.is_constant(ModuleNamespace)
    assert name.is_constant(str)
    resolved = ns.value[name.value]
    ct = Constant(resolved)
    yield Replace(ct)
    res = yield Require(ct)
    return res


def user_switch(node, unif):
    """Inferrer for the user_switch basic function."""
    cond, ift, iff = node.inputs
    _ = yield Require(cond)  # TODO: check bool
    ift_t = yield Require(ift)
    iff_t = yield Require(iff)
    return data.AbstractUnion([ift_t, iff_t], tracks={})


def getattr_inferrer(node, unif):
    """Inferrer for the getattr function."""
    obj_node, key_node = node.inputs
    assert key_node.is_constant(str)
    obj = yield Require(obj_node)
    key = key_node.value
    result = getattr(obj.tracks.interface, key)
    ct = Constant(result)
    yield Replace(ct)
    res = yield Require(ct)
    return res


X = data.Generic("x")


def add_standard_inferrers(inferrers):
    """Register all the inferrers in this file."""
    inferrers.update(
        {
            operator.mul: signature(X, X, ret=X),
            operator.add: signature(X, X, ret=X),
            operator.sub: signature(X, X, ret=X),
            operator.neg: signature(X, ret=X),
            operator.le: signature(X, X, ret=bool),
            operator.truth: signature(X, ret=bool),
            basics.return_: signature(X, ret=X),
            basics.resolve: inference_function(resolve),
            basics.user_switch: inference_function(user_switch),
            int.__add__: signature(int, int, ret=int),
            float.__add__: signature(float, float, ret=float),
            getattr: inference_function(getattr_inferrer),
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
        }
    )
