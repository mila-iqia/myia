"""Inferrers for basic functions and the standard library."""

import operator
import types

from .. import basics
from ..abstract import data, utils as autils
from ..abstract.to_abstract import precise_abstract
from ..basics import Handle
from ..ir import Constant
from ..utils.misc import ModuleNamespace
from .algo import Require, RequireAll
from .infnode import (
    InferenceEngine,
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
    assert key_node.is_constant(
        str
    ), f"getattr: expected a constant string key, got {key_node}"
    obj = yield Require(obj_node)
    if InferenceEngine.is_abstract_type(obj):
        obj = obj.elements[0]
    key = key_node.value
    interface = obj.tracks.interface
    result = getattr(interface, key)
    if isinstance(result, (types.MethodType, types.WrapperDescriptorType)):
        ct = Constant(result)
        new_node = node.graph.apply(basics.partial, ct, obj_node)
    elif isinstance(interface, types.ModuleType) and isinstance(
        result, (types.FunctionType, types.BuiltinFunctionType)
    ):
        new_node = Constant(result)
    else:
        raise AssertionError(
            f"getattr can currently only be used for methods, got {interface}.{result} (type {type(result)})"
        )
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

    if not (
        isinstance(obj, data.AbstractStructure)
        and obj.tracks.interface in (tuple, list)
    ):
        raise AssertionError(
            f"getitem can currently only be used for "
            f"dicts, lists and tuples, got {obj}"
        )

    if key_node.is_apply(slice):
        if obj.tracks.interface is list:
            return data.AbstractStructure(
                [obj.elements[0]], {"interface": list}
            )
        elif all(inp.is_constant() for inp in key_node.inputs):
            idx = slice(*(inp.value for inp in key_node.inputs))
            selection = obj.elements[idx]
            return data.AbstractStructure(selection, {"interface": tuple})
        else:
            raise AssertionError(
                "getitem inferrer does not yet support non-constants slice for tuples"
            )
    else:
        assert key_node.is_constant(int), key_node
        if obj.tracks.interface is list:
            return obj.elements[0]
        else:
            key = key_node.value
            return obj.elements[key]


def make_tuple_inferrer(node, args, unif):
    """Inferrer for the make_tuple function."""
    tuple_types = yield RequireAll(*args)
    return data.AbstractStructure(tuple_types, {"interface": tuple})


def make_list_inferrer(node, args, unif):
    """Inferrer for the make_list function."""
    elements = []
    if args:
        tuple_types = yield RequireAll(*args)
        base_type = tuple_types[0]
        for typ in tuple_types[1:]:
            base_type = autils.merge(base_type, typ, U=unif)[0]
        elements.append(base_type)
    return data.AbstractStructure(elements, {"interface": list})


def make_dict_inferrer(node, args, unif):
    """Inferrer for the make_dict function."""
    arg_types = yield RequireAll(*args)
    return data.AbstractDict(arg_types)


def len_inferrer(node, args, unif):
    """Inferrer for the len function."""
    (obj_node,) = args
    obj_type = yield Require(obj_node)
    interface = obj_type.tracks.interface
    if not hasattr(interface, "__len__"):
        raise AttributeError(f"Interface has no attribute __len__: {interface}")
    return data.AbstractAtom({"interface": int})


def isinstance_inferrer(node, args, unif):
    """Inferrer for the isinstance function."""
    obj_node, cls_node = args
    obj_type = yield Require(obj_node)
    cls_type = yield Require(cls_node)

    if isinstance(obj_type, data.AbstractUnion):
        inp_types = []
        for el in obj_type.options:
            el_interface = el.tracks.interface
            assert isinstance(el_interface, type), el_interface
            inp_types.append(el_interface)
    else:
        obj_cls = obj_type.tracks.interface
        assert isinstance(obj_cls, type), obj_cls
        inp_types = [obj_cls]

    if cls_type.tracks.interface is tuple:
        assert isinstance(cls_type, data.AbstractStructure)
        out_types = []
        for el in cls_type.elements:
            assert InferenceEngine.is_abstract_type(
                el
            ), f"Expected abstract type, got {el}"
            el_interface = el.elements[0].tracks.interface
            assert isinstance(el_interface, type), el_interface
            out_types.append(el_interface)
    else:
        assert InferenceEngine.is_abstract_type(
            cls_type
        ), f"Expected abstract type, got {cls_type}"
        expected_type = cls_type.elements[0]
        if isinstance(expected_type, data.AbstractUnion):
            out_types = []
            for el in expected_type.options:
                el_interface = el.tracks.interface
                assert isinstance(el_interface, type), el_interface
                out_types.append(el_interface)
        else:
            expected_cls = expected_type.tracks.interface
            assert isinstance(expected_cls, type), expected_cls
            out_types = [expected_cls]
    assert (
        object not in out_types
    ), "Too broad type `object` expected for isinstance"

    expected = tuple(out_types)
    return precise_abstract(any(issubclass(el, expected) for el in inp_types))


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
            [type_el, data.AbstractStructure([type_el], {"interface": list})],
            {"interface": tuple},
        )


def _unary_op_inferrer(unary_op, node, args, unif):
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


def _bin_op_inferrer(bin_op, bin_rop, node, args, unif):
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
    elif hasattr(b_interface, bin_rop):
        new_node = node.graph.apply(
            getattr(b_interface, bin_rop), b_node, a_node
        )
    else:
        raise TypeError(
            f"No {bin_op} method for {a_interface} and no {bin_rop} method for {b_interface}"
        )
    yield Replace(new_node)
    res = yield Require(new_node)
    return res


def _bin_rop_inferrer(bin_rop, node, args, unif):
    """Generic inferrer for binary rop methods.

    Replace node with a call to rop method if available in right operand type.
    """
    a_node, b_node = args
    b_type = yield Require(b_node)
    b_interface = b_type.tracks.interface
    if hasattr(b_interface, bin_rop):
        new_node = node.graph.apply(
            getattr(b_interface, bin_rop), b_node, a_node
        )
        yield Replace(new_node)
        res = yield Require(new_node)
        return res
    else:
        raise TypeError(f"No {bin_rop} method for {b_interface}")


def _bin_op_dispatcher(bin_rop, *signatures):
    """Dispatcher for binary op methods.

    Create an inference function using given signatures and a
    fallback binary rop inference function.
    """

    def inf(node, args, unif):
        return _bin_rop_inferrer(bin_rop, node, args, unif)

    return dispatch_inferences(*signatures, default=inf)


def _bin_op_cast_right(cls, bin_op, node, args, unif):
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
        return data.AbstractAtom({"interface": cls})
    else:
        b_casted = node.graph.apply(cls, b_node)
        new_node = node.graph.apply(getattr(cls, bin_op), a_node, b_casted)
        yield Replace(new_node)
        res = yield Require(new_node)
        return res


def operator_add_inferrer(node, args, unif):
    """Inferrer for the operator.add function."""
    return _bin_op_inferrer("__add__", "__radd__", node, args, unif)


def operator_floordiv_inferrer(node, args, unif):
    """Inferrer for the operator.floordiv function."""
    return _bin_op_inferrer("__floordiv__", "__rfloordiv__", node, args, unif)


def operator_mod_inferrer(node, args, unif):
    """Inferrer for the operator.mod function."""
    return _bin_op_inferrer("__mod__", "__rmod__", node, args, unif)


def operator_sub_inferrer(node, args, unif):
    """Inferrer for the operator.sub function."""
    return _bin_op_inferrer("__sub__", "__rsub__", node, args, unif)


def operator_truediv_inferrer(node, args, unif):
    """Inferrer for the operator.truediv function."""
    return _bin_op_inferrer("__truediv__", "__rtruediv__", node, args, unif)


def operator_mul_inferrer(node, args, unif):
    """Inferrer for the operator.mul function."""
    return _bin_op_inferrer("__mul__", "__rmul__", node, args, unif)


def operator_neg_inferrer(node, args, unif):
    """Inferrer for the operator.neg function."""
    return _unary_op_inferrer("__neg__", node, args, unif)


def operator_pos_inferrer(node, args, unif):
    """Inferrer for the operator.pos function."""
    return _unary_op_inferrer("__pos__", node, args, unif)


def operator_pow_inferrer(node, args, unif):
    """Inferrer for the operator.pow function."""
    return _bin_op_inferrer("__pow__", "__rpow__", node, args, unif)


def float_mul_inferrer(node, args, unif):
    """Inferrer for the float.__mul__ function."""
    return _bin_op_cast_right(float, "__mul__", node, args, unif)


def float_rmul_inferrer(node, args, unif):
    """Inferrer for the float.__mul__ function."""
    return _bin_op_cast_right(float, "__rmul__", node, args, unif)


def float_radd_inferrer(node, args, unif):
    """Inferrer for the float.__radd__ function."""
    return _bin_op_cast_right(float, "__radd__", node, args, unif)


def float_sub_inferrer(node, args, unif):
    """Inferrer for the float.__sub__ function."""
    return _bin_op_cast_right(float, "__sub__", node, args, unif)


def int_neg_inferrer(node, args, unif):
    """Inferrer for the int.__neg__ function."""
    # NB: bool.__neg__ is int.__neg__ and return an int.
    # So, we must expect input type to be either int or bool.
    (value_node,) = args
    if value_node.is_constant((int, bool)):
        ct = Constant(-value_node.value)
        yield Replace(ct)
        res = yield Require(ct)
        return res
    else:
        value_type = yield Require(value_node)
        assert value_type.tracks.interface in (int, bool), value_type
        return data.AbstractAtom({"interface": int})


def tuple_add_inferrer(node, args, unif):
    """Inferrer for the tuple.__add__ function."""
    t1_node, t2_node = args
    t1_type = yield Require(t1_node)
    t2_type = yield Require(t2_node)
    assert isinstance(
        t1_type, data.AbstractStructure
    ), f"Expected abstract tuple, got {t1_type}"
    assert isinstance(
        t2_type, data.AbstractStructure
    ), f"Expected abstract tuple, got {t2_type}"
    assert t1_type.tracks.interface is tuple
    assert t2_type.tracks.interface is tuple
    return data.AbstractStructure(
        t1_type.elements + t2_type.elements, {"interface": tuple}
    )


X = data.Generic("x")
Y = data.Generic("y")


def add_standard_inferrers(inferrers):
    """Register all the inferrers in this file."""
    inferrers.update(
        {
            # operator functions
            operator.add: inference_function(operator_add_inferrer),
            operator.and_: signature(X, X, ret=X),
            operator.eq: signature(X, Y, ret=bool),
            operator.floordiv: inference_function(operator_floordiv_inferrer),
            operator.getitem: inference_function(getitem_inferrer),
            operator.gt: signature(X, Y, ret=bool),
            operator.invert: signature(X, ret=X),
            operator.is_: signature(X, Y, ret=bool),
            operator.is_not: signature(X, Y, ret=bool),
            operator.le: signature(X, X, ret=bool),
            operator.lshift: signature(X, X, ret=X),
            operator.lt: signature(X, X, ret=bool),
            operator.mod: inference_function(operator_mod_inferrer),
            operator.mul: inference_function(operator_mul_inferrer),
            operator.ne: signature(X, X, ret=X),
            operator.neg: inference_function(operator_neg_inferrer),
            operator.not_: signature(X, ret=bool),
            operator.or_: signature(X, X, ret=X),
            operator.pos: inference_function(operator_pos_inferrer),
            operator.pow: inference_function(operator_pow_inferrer),
            operator.rshift: signature(X, X, ret=X),
            operator.sub: inference_function(operator_sub_inferrer),
            operator.truediv: inference_function(operator_truediv_inferrer),
            operator.truth: signature(X, ret=bool),
            operator.xor: signature(X, X, ret=X),
            basics.return_: signature(X, ret=X),
            basics.resolve: inference_function(resolve),
            basics.user_switch: inference_function(user_switch),
            # builtin functions
            len: inference_function(len_inferrer),
            hasattr: signature(X, str, ret=bool),
            isinstance: inference_function(isinstance_inferrer),
            int.__add__: _bin_op_dispatcher("__radd__", (int, int, int)),
            int.__mul__: _bin_op_dispatcher(
                "__rmul__",
                (int, int, int),
                (bool, bool, int),
            ),
            int.__sub__: signature(int, int, ret=int),
            int.__neg__: inference_function(int_neg_inferrer),
            float.__add__: signature(float, float, ret=float),
            float.__neg__: signature(X, ret=X),
            float.__mul__: inference_function(float_mul_inferrer),
            float.__sub__: inference_function(float_sub_inferrer),
            float.__radd__: inference_function(float_radd_inferrer),
            float.__rmul__: inference_function(float_rmul_inferrer),
            tuple.__add__: inference_function(tuple_add_inferrer),
            getattr: inference_function(getattr_inferrer),
            type: signature(
                X,
                ret=data.AbstractStructure([X], tracks={"interface": type}),
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
            basics.make_dict: inference_function(make_dict_inferrer),
            basics.make_list: inference_function(make_list_inferrer),
            basics.make_tuple: inference_function(make_tuple_inferrer),
            basics.myia_hasnext: inference_function(myia_hasnext_inferrer),
            basics.myia_iter: inference_function(myia_iter_inferrer),
            basics.myia_next: inference_function(myia_next_inferrer),
            basics.partial: inference_function(partial_inferrer),
        }
    )
