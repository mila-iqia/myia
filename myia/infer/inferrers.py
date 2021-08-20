"""Inferrers for basic functions and the standard library."""

import operator
import types

from .. import basics
from ..abstract import data, utils as autils
from ..basics import Handle
from ..ir import Constant
from ..utils.misc import ModuleNamespace
from .algo import Require, RequireAll
from .infnode import (
    Replace,
    dispatch_inferences,
    inference_function,
    signature,
)


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
    if isinstance(interface, data.TypedObject):
        if key in interface.indexed:
            return obj.elements[interface.indexed[key]]
        else:
            interface = interface.cls
    result = getattr(interface, key)
    if isinstance(interface, types.ModuleType) and isinstance(
        result, (types.FunctionType, types.BuiltinFunctionType)
    ):
        new_node = Constant(result)
    elif isinstance(
        result,
        (types.MethodType, types.WrapperDescriptorType, types.FunctionType),
    ):
        ct = Constant(result)
        new_node = node.graph.apply(basics.partial, ct, obj_node)
    else:
        raise AssertionError("getattr can currently only be used for methods")
        # new_node = Constant(result)
    yield Replace(new_node)
    res = yield Require(new_node)
    return res


def _unary_op_inference(unary_op):
    """Create a generic inference for builtin unary operator functions."""

    def inf(node, args, unif):
        """Generic inferrer for builtin unary operator functions."""
        (a_node,) = args
        a_type = yield Require(a_node)
        a_interface = a_type.tracks.interface
        if not hasattr(a_interface, unary_op):
            raise TypeError(f"No {unary_op} method for type {a_interface}")
        new_node = node.graph.apply(getattr(a_interface, unary_op), a_node)
        yield Replace(new_node)
        res = yield Require(new_node)
        return res

    return inference_function(inf)


def _bin_op_inference(bin_op, bin_rop):
    """Create a generic inference for builtin binary operator functions."""

    def inf(node, args, unif):
        """Generic inferrer for builtin binary operator functions."""
        a_node, b_node = args
        a_type = yield Require(a_node)
        b_type = yield Require(b_node)

        if isinstance(a_type, data.GenericBase) or isinstance(
            b_type, data.GenericBase
        ):
            # If there is any generic in operands, just unify them.
            return autils.unify(a_type, b_type)[0]

        a_interface = a_type.tracks.interface
        b_interface = b_type.tracks.interface
        if hasattr(a_interface, bin_op):
            new_node = node.graph.apply(
                getattr(a_interface, bin_op), a_node, b_node
            )
        # TODO: In order to support user-defined graphs for bin_op,
        # the case where it returns NotImplemented will have to be handled.
        elif hasattr(b_interface, bin_rop):
            new_node = node.graph.apply(
                getattr(b_interface, bin_rop), b_node, a_node
            )
        else:
            raise TypeError(
                f"No {bin_op} method for {a_interface} "
                f"and no {bin_rop} method for {b_interface}"
            )
        yield Replace(new_node)
        res = yield Require(new_node)
        return res

    return inference_function(inf)


def _bin_rop_inferrer(bin_rop):
    """Create a generic inferrer for binary rop methods."""

    def inf(node, args, unif):
        """Generic inferrer for binary rop methods.

        Replace node with a call to rop method
        if available in right operand type.
        """
        a_node, b_node = args
        b_type = yield Require(b_node)
        b_interface = b_type.tracks.interface
        assert hasattr(
            b_interface, bin_rop
        ), f"No {bin_rop} method for {b_interface}"
        new_node = node.graph.apply(
            getattr(b_interface, bin_rop), b_node, a_node
        )
        yield Replace(new_node)
        res = yield Require(new_node)
        return res

    return inf


def _bin_op_dispatcher(bin_rop, *signatures):
    """Dispatcher for binary op methods.

    Create an inference function using given signatures and a
    fallback binary rop inference function.
    """

    return dispatch_inferences(*signatures, default=_bin_rop_inferrer(bin_rop))


def _bin_op_right_cast_inference(cls, bin_op, ret=None):
    """Create a generic inference for binary op/rop methods.

    Replace right operand with a cast to left type if necessary.

    If both operands have same interface cls,
    return abstract atom with either cls or ret (if not None) as interface.
    """
    if ret is None:
        ret = cls
    else:
        assert isinstance(ret, type)

    def inf(node, args, unif):
        """Inferrer for binary op/rop methods (e.g. float.__mul__).

        Cast right operand to left operand type if necessary.
        """
        a_node, b_node = args
        a_type = yield Require(a_node)
        b_type = yield Require(b_node)
        a_interface = a_type.tracks.interface
        b_interface = b_type.tracks.interface
        assert a_interface is cls, f"expected {cls}, got {a_interface}"
        if b_interface is cls:
            return data.AbstractAtom({"interface": ret})
        else:
            b_casted = node.graph.apply(cls, b_node)
            new_node = node.graph.apply(getattr(cls, bin_op), a_node, b_casted)
            yield Replace(new_node)
            res = yield Require(new_node)
            return res

    return inference_function(inf)


X = data.Generic("x")
Y = data.Generic("y")


def add_standard_inferrers(inferrers):
    """Register all the inferrers in this file."""
    inferrers.update(
        {
            # operator functions
            operator.add: _bin_op_inference("__add__", "__radd__"),
            operator.and_: _bin_op_inference("__and__", "__rand__"),
            operator.eq: _bin_op_inference("__eq__", "__eq__"),
            operator.floordiv: _bin_op_inference(
                "__floordiv__", "__rfloordiv__"
            ),
            operator.ge: _bin_op_inference("__ge__", "__le__"),
            operator.gt: _bin_op_inference("__gt__", "__lt__"),
            operator.invert: _unary_op_inference("__invert__"),
            operator.is_: signature(X, Y, ret=bool),
            operator.is_not: signature(X, Y, ret=bool),
            operator.le: _bin_op_inference("__le__", "__ge__"),
            operator.lshift: _bin_op_inference("__lshift__", "__rlshift__"),
            operator.lt: _bin_op_inference("__lt__", "__gt__"),
            operator.mod: _bin_op_inference("__mod__", "__rmod__"),
            operator.mul: _bin_op_inference("__mul__", "__rmul__"),
            operator.ne: _bin_op_inference("__ne__", "__ne__"),
            operator.neg: signature(X, ret=X),
            operator.not_: signature(X, ret=bool),
            operator.or_: _bin_op_inference("__or__", "__ror__"),
            operator.pos: _unary_op_inference("__pos__"),
            operator.pow: _bin_op_inference("__pow__", "__rpow__"),
            operator.rshift: _bin_op_inference("__rshift__", "__rrshift__"),
            operator.sub: _bin_op_inference("__sub__", "__rsub__"),
            operator.truediv: _bin_op_inference("__truediv__", "__rtruediv__"),
            operator.truth: signature(X, ret=bool),
            operator.xor: _bin_op_inference("__xor__", "__rxor__"),
            # builtin constructors
            type(None): signature(ret=None),
            bool: dispatch_inferences(
                (bool, bool),
                (int, bool),
                (float, bool),
            ),
            int: dispatch_inferences(
                (bool, int),
                (int, int),
                (float, int),
            ),
            float: dispatch_inferences(
                (bool, float),
                (int, float),
                (float, float),
            ),
            # builtin functions
            getattr: inference_function(getattr_inferrer),
            type: signature(
                X,
                ret=data.AbstractStructure([X], tracks={"interface": type}),
            ),
            bool.__add__: _bin_op_dispatcher("__radd__", (bool, bool, bool)),
            bool.__and__: dispatch_inferences(
                (bool, bool, bool),
                (bool, int, int),
            ),
            bool.__or__: dispatch_inferences(
                (bool, bool, bool),
                (bool, int, int),
            ),
            bool.__xor__: dispatch_inferences(
                (bool, bool, bool),
                (bool, int, int),
            ),
            float.__add__: _bin_op_right_cast_inference(float, "__add__"),
            float.__eq__: _bin_op_right_cast_inference(float, "__eq__", bool),
            float.__floordiv__: _bin_op_right_cast_inference(
                float, "__floordiv__"
            ),
            float.__ge__: _bin_op_right_cast_inference(float, "__ge__", bool),
            float.__gt__: _bin_op_right_cast_inference(float, "__gt__", bool),
            float.__le__: _bin_op_right_cast_inference(float, "__le__", bool),
            float.__lt__: _bin_op_right_cast_inference(float, "__lt__", bool),
            float.__mod__: dispatch_inferences(
                (float, bool, float),
                (float, int, float),
                (float, float, float),
            ),
            float.__mul__: _bin_op_right_cast_inference(float, "__mul__"),
            float.__ne__: _bin_op_right_cast_inference(float, "__ne__", bool),
            float.__pos__: signature(float, ret=float),
            float.__pow__: dispatch_inferences(
                (float, bool, float),
                (float, int, float),
                (float, float, float),
            ),
            float.__radd__: _bin_op_right_cast_inference(float, "__radd__"),
            float.__rfloordiv__: _bin_op_right_cast_inference(
                float, "__rfloordiv__"
            ),
            float.__rmod__: _bin_op_right_cast_inference(float, "__rmod__"),
            float.__rmul__: _bin_op_right_cast_inference(float, "__rmul__"),
            float.__rpow__: _bin_op_right_cast_inference(float, "__rpow__"),
            float.__rsub__: _bin_op_right_cast_inference(float, "__rsub__"),
            float.__sub__: _bin_op_right_cast_inference(float, "__sub__"),
            float.__truediv__: _bin_op_right_cast_inference(
                float, "__truediv__"
            ),
            int.__add__: _bin_op_dispatcher(
                "__radd__", (int, int, int), (bool, bool, int), (int, bool, int)
            ),
            int.__and__: dispatch_inferences(
                (int, bool, int),
                (int, int, int),
            ),
            int.__eq__: _bin_op_dispatcher(
                "__eq__",
                (bool, bool, bool),
                (bool, int, bool),
                (int, bool, bool),
                (int, int, bool),
            ),
            int.__floordiv__: _bin_op_dispatcher(
                "__rfloordiv__",
                (bool, bool, int),
                (bool, int, int),
                (int, bool, int),
                (int, int, int),
            ),
            int.__ge__: _bin_op_dispatcher(
                "__le__",
                (bool, bool, bool),
                (bool, int, bool),
                (int, bool, bool),
                (int, int, bool),
            ),
            int.__gt__: _bin_op_dispatcher(
                "__lt__",
                (bool, bool, bool),
                (bool, int, bool),
                (int, bool, bool),
                (int, int, bool),
            ),
            int.__invert__: dispatch_inferences(
                (int, int),
                (bool, int),
            ),
            int.__le__: _bin_op_dispatcher(
                "__ge__",
                (bool, bool, bool),
                (bool, int, bool),
                (int, bool, bool),
                (int, int, bool),
            ),
            int.__lshift__: dispatch_inferences(
                (bool, bool, int),
                (bool, int, int),
                (int, bool, int),
                (int, int, int),
            ),
            int.__lt__: _bin_op_dispatcher(
                "__gt__",
                (bool, bool, bool),
                (bool, int, bool),
                (int, bool, bool),
                (int, int, bool),
            ),
            int.__mod__: _bin_op_dispatcher(
                "__rmod__",
                (bool, bool, int),
                (bool, int, int),
                (int, bool, int),
                (int, int, int),
            ),
            int.__mul__: _bin_op_dispatcher(
                "__rmul__",
                (bool, bool, int),
                (bool, int, int),
                (int, bool, int),
                (int, int, int),
            ),
            int.__ne__: _bin_op_dispatcher(
                "__ne__",
                (bool, bool, bool),
                (bool, int, bool),
                (int, bool, bool),
                (int, int, bool),
            ),
            int.__or__: dispatch_inferences(
                (int, bool, int),
                (int, int, int),
            ),
            int.__pos__: dispatch_inferences(
                (int, int),
                (bool, int),
            ),
            int.__pow__: _bin_op_dispatcher(
                "__rpow__",
                (bool, bool, int),
                (bool, int, int),
                (int, bool, int),
                (int, int, int),
            ),
            int.__radd__: dispatch_inferences((int, bool, int)),
            int.__rshift__: dispatch_inferences(
                (bool, bool, int),
                (bool, int, int),
                (int, bool, int),
                (int, int, int),
            ),
            int.__sub__: _bin_op_dispatcher(
                "__rsub__",
                (bool, bool, int),
                (bool, int, int),
                (int, bool, int),
                (int, int, int),
            ),
            int.__truediv__: dispatch_inferences(
                (bool, bool, float),
                (bool, int, float),
                (bool, float, float),
                (int, bool, float),
                (int, int, float),
                (int, float, float),
            ),
            int.__xor__: dispatch_inferences(
                (int, bool, int),
                (int, int, int),
            ),
            # NB: For str comparison methods,
            # we declare str-vs-other-builtin signatures as not implemented
            # so that inference stops here instead of looking for
            # right operand method (which might call back
            # str comparison method again, resulting in infinite loop).
            str.__eq__: _bin_op_dispatcher(
                "__eq__",
                (str, bool, NotImplemented),
                (str, int, NotImplemented),
                (str, float, NotImplemented),
                (str, str, bool),
            ),
            str.__ge__: _bin_op_dispatcher(
                "__le__",
                (str, bool, NotImplemented),
                (str, int, NotImplemented),
                (str, float, NotImplemented),
                (str, str, bool),
            ),
            str.__gt__: _bin_op_dispatcher(
                "__lt__",
                (str, bool, NotImplemented),
                (str, int, NotImplemented),
                (str, float, NotImplemented),
                (str, str, bool),
            ),
            str.__le__: _bin_op_dispatcher(
                "__ge__",
                (str, bool, NotImplemented),
                (str, int, NotImplemented),
                (str, float, NotImplemented),
                (str, str, bool),
            ),
            str.__lt__: _bin_op_dispatcher(
                "__gt__",
                (str, bool, NotImplemented),
                (str, int, NotImplemented),
                (str, float, NotImplemented),
                (str, str, bool),
            ),
            str.__ne__: _bin_op_dispatcher(
                "__ne__",
                (str, bool, NotImplemented),
                (str, int, NotImplemented),
                (str, float, NotImplemented),
                (str, str, bool),
            ),
            # myia basics functions
            basics.global_universe_getitem: signature(
                data.AbstractStructure([X], tracks={"interface": Handle}),
                ret=X,
            ),
            basics.global_universe_setitem: signature(
                data.AbstractStructure([X], tracks={"interface": Handle}),
                X,
                ret=None,
            ),
            basics.make_handle: signature(
                data.AbstractStructure([X], tracks={"interface": type}),
                ret=data.AbstractStructure([X], tracks={"interface": Handle}),
            ),
            basics.partial: inference_function(partial_inferrer),
            basics.resolve: inference_function(resolve),
            basics.return_: signature(X, ret=X),
            basics.user_switch: inference_function(user_switch),
        }
    )
