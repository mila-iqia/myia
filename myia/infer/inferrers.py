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
from .infnode import Replace, inference_function, signature, InferenceEngine


def resolve(node, args, unif, inferrers):
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


def user_switch(node, args, unif, inferrers):
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


def partial_inferrer(node, args, unif, inferrers):
    """Inferrer for the a partial application."""
    fn, *args = args
    fn_type = yield Require(fn)
    arg_types = yield RequireAll(*args)
    return data.AbstractStructure(
        (*fn_type.elements, *arg_types),
        tracks={"interface": fn_type.tracks.interface},
    )


def getattr_inferrer(node, args, unif, inferrers):
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


def getitem_inferrer(node, args, unif, inferrers):
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


def make_tuple_inferrer(node, args, unif, inferrers):
    """Inferrer for the make_tuple function."""
    tuple_types = []
    for arg_node in args:
        tuple_types.append((yield Require(arg_node)))
    return data.AbstractStructure(tuple_types, {"interface": tuple})


def make_list_inferrer(node, args, unif, inferrers):
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


def make_dict_inferrer(node, args, unif, inferrers):
    """Inferrer for the make_dict function."""
    assert (
        not len(args) % 2
    ), f"make_dict: expected even number of arguments, got {len(args)}"
    arg_types = yield RequireAll(*args)
    return data.AbstractDict(arg_types)


def len_inferrer(node, args, unif, inferrers):
    obj_node, = args
    obj_type = yield Require(obj_node)
    interface = obj_type.tracks.interface
    if not hasattr(interface, "__len__"):
        raise AttributeError(f"Interface has no attribute __len__: {interface}")
    return data.AbstractAtom({"interface": int})


def isinstance_inferrer(node, args, unif, inferrers):
    obj_node, cls_node = args
    obj_type = yield Require(obj_node)
    cls_type = yield Require(cls_node)

    obj_cls = obj_type.tracks.interface
    assert isinstance(obj_cls, type), obj_cls

    if cls_type.tracks.interface is tuple:
        assert isinstance(cls_type, data.AbstractStructure)
        expected_classes = []
        for el in cls_type.elements:
            assert InferenceEngine.is_abstract_type(el)
            expected_classes.append(el.elements[0].tracks.interface)
    else:
        assert InferenceEngine.is_abstract_type(cls_type), f"Expected abstract value, got {cls_type}"
        expected_classes = [cls_type.elements[0].tracks.interface]

    assert all(isinstance(cls, type) for cls in expected_classes), expected_classes
    # print("testing", obj_cls, expected_classes)
    return precise_abstract(issubclass(obj_cls, tuple(expected_classes)))


def myia_iter_inferrer(node, args, unif, inferrers):
    """Inferrer for the myia_iter function."""
    (iterable_node,) = args
    iterable_type = yield Require(iterable_node)
    if iterable_type.tracks.interface in (tuple, list, range):
        return iterable_type
    raise TypeError(f"myia_iter: unexpected input type: {iterable_type}")


def myia_hasnext_inferrer(node, args, unif, inferrers):
    """Inferrer for the myia_hasnext function."""
    (iterable_node,) = args
    iterable_type = yield Require(iterable_node)
    if iterable_type.tracks.interface in (range, list):
        return data.AbstractAtom({"interface": bool})
    if iterable_type.tracks.interface is tuple:
        assert isinstance(iterable_type, data.AbstractStructure)
        return precise_abstract(bool(iterable_type.elements))
    raise TypeError(f"myia_hasnext: unexpected input type: {iterable_type}")


def myia_next_inferrer(node, args, unif, inferrers):
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


def _bin_op_inferrer(bin_op, node, args, unif, inferrers):
    a_node, b_node = args
    a_type = yield Require(a_node)
    b_type = yield Require(b_node)
    if isinstance(a_type, data.AbstractValue):
        a_interface = a_type.tracks.interface
    else:
        a_interface = a_type
    if isinstance(b_type, data.AbstractValue):
        b_interface = b_type.tracks.interface
    else:
        b_interface = b_type
    if a_interface is b_interface:
        if not hasattr(a_interface, bin_op):
            raise TypeError(f"No {bin_op} method for type {a_interface}")
        elif getattr(a_interface, bin_op) in inferrers:
            res = inferrers[getattr(a_interface, bin_op)].tracks.interface.fn(node, args, unif, inferrers)
        else:
            # Assume binary op on same type return same type
            return data.AbstractAtom({"interface": a_interface})
    elif hasattr(a_interface, bin_op) and getattr(a_interface, bin_op) in inferrers:
        res = inferrers[getattr(a_interface, bin_op)].tracks.interface.fn(node, args, unif, inferrers)
    else:
        raise TypeError(f"No {bin_op} inference for {a_interface} + {b_interface}")
    if isinstance(res, types.GeneratorType):
        assert isinstance(res, types.GeneratorType)
        curr = None
        try:
            while True:
                instruction = res.send(curr)
                if isinstance(instruction, Replace):
                    node.replace(instruction.new_node)
                    curr = None
                else:
                    curr = yield instruction
        except StopIteration as stop:
            return stop.value
    else:
        return res


def _unary_op_inferrer(unary_op, node, args, unif, inferrers):
    a_node, = args
    a_type = yield Require(a_node)
    if isinstance(a_type, data.AbstractValue):
        a_interface = a_type.tracks.interface
    else:
        a_interface = a_type

    if not hasattr(a_interface, unary_op):
        raise TypeError(f"No {unary_op} method for type {a_interface}")
    elif getattr(a_interface, unary_op) not in inferrers:
        # Assume unary op return same type
        return data.AbstractAtom({"interface": a_interface})
    else:
        res = inferrers[getattr(a_interface, unary_op)].tracks.interface.fn(node, args, unif, inferrers)
        if isinstance(res, types.GeneratorType):
            assert isinstance(res, types.GeneratorType)
            curr = None
            try:
                while True:
                    instruction = res.send(curr)
                    if isinstance(instruction, Replace):
                        node.replace(instruction.new_node)
                        curr = None
                    else:
                        curr = yield instruction
            except StopIteration as stop:
                return stop.value
        else:
            return res


def operator_add_inferrer(node, args, unif, inferrers):
    return _bin_op_inferrer("__add__", node, args, unif, inferrers)


def operator_floordiv_inferrer(node, args, unif, inferrers):
    return _bin_op_inferrer("__floordiv__", node, args, unif, inferrers)


def operator_mod_inferrer(node, args, unif, inferrers):
    return _bin_op_inferrer("__mod__", node, args, unif, inferrers)


def operator_sub_inferrer(node, args, unif, inferrers):
    return _bin_op_inferrer("__sub__", node, args, unif, inferrers)


def operator_truediv_inferrer(node, args, unif, inferrers):
    return _bin_op_inferrer("__truediv__", node, args, unif, inferrers)


def operator_mul_inferrer(node, args, unif, inferrers):
    return _bin_op_inferrer("__mul__", node, args, unif, inferrers)


def operator_neg_inferrer(node, args, unif, inferrers):
    return _unary_op_inferrer("__neg__", node, args, unif, inferrers)


def operator_pos_inferrer(node, args, unif, inferrers):
    return _unary_op_inferrer("__pos__", node, args, unif, inferrers)


def operator_pow_inferrer(node, args, unif, inferrers):
    return _bin_op_inferrer("__pow__", node, args, unif, inferrers)


def float_sub_inferrer(node, args, unif, inferrers):
    a_node, b_node = args
    a_interface = yield Require(a_node)
    b_interface = yield Require(b_node)
    if isinstance(a_interface, data.AbstractValue):
        a_interface = a_interface.tracks.interface
    if isinstance(b_interface, data.AbstractValue):
        b_interface = b_interface.tracks.interface
    assert a_interface is float, f"expected float type, got {a_interface}"
    if b_interface is float:
        return data.AbstractAtom({"interface": float})
    else:
        b_casted = node.graph.apply(float, b_node)
        new_node = node.graph.apply(float.__sub__, a_node, b_casted)
        yield Replace(new_node)
        res = yield Require(new_node)
        return res


def tuple_add_inferrer(node, args, unif, inferrers):
    t1_node, t2_node = args
    t1_type = yield Require(t1_node)
    t2_type = yield Require(t2_node)
    assert isinstance(t1_type, data.AbstractStructure), f"Expected abstract tuple, got {t1_type}"
    assert isinstance(t2_type, data.AbstractStructure), f"Expected abstract tuple, got {t2_type}"
    assert t1_type.tracks.interface is tuple
    assert t2_type.tracks.interface is tuple
    return data.AbstractStructure(t1_type.elements + t2_type.elements, {"interface": tuple})


X = data.Generic("x")
Y = data.Generic("y")


def add_standard_inferrers(inferrers):
    """Register all the inferrers in this file."""
    inferrers.update(
        {
            # operator functions
            operator.add: inference_function(operator_add_inferrer),
            operator.and_: signature(X, X, ret=X),
            operator.eq: signature(X, X, ret=bool),
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
            int.__add__: signature(int, int, ret=int),
            float.__add__: signature(float, float, ret=float),
            float.__sub__: inference_function(float_sub_inferrer),
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
