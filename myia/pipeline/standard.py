"""Pre-made pipelines."""

import builtins
import math
import operator

import numpy as np

from .. import operations, xtype
from ..abstract import ABSENT, Context
from ..ir import GraphManager
from ..operations import primitives as P
from ..operations.gen import lop, reverse_binop, rop
from ..utils import Registry
from ..validate import (
    validate_abstract as default_validate_abstract,
    whitelist as default_whitelist,
)
from . import steps
from .pipeline import PipelineDefinition
from .resources import (
    BackendResource,
    ConverterResource,
    DebugVMResource,
    InferenceResource,
    Resources,
)

py_registry = Registry(default_field='python_implementation')
vm_registry = Registry(default_field='debugvm_implementation')
grad_registry = Registry(default_field='grad_transform')
inferrer_registry = Registry(default_field='inferrer_constructor')


python_operation_map = {
    builtins.Exception: operations.make_exception,
    builtins.bool: operations.bool,
    builtins.getattr: operations.getattr,
    builtins.enumerate: operations.enumerate,
    builtins.hasattr: operations.hasattr,
    builtins.isinstance: operations.isinstance,
    builtins.len: operations.len,
    builtins.range: operations.range,
    builtins.slice: operations.slice,
    builtins.sum: operations.sum,
    builtins.zip: operations.zip,
    operator.add: operations.add,
    operator.sub: operations.sub,
    operator.mul: operations.mul,
    operator.truediv: operations.truediv,
    operator.floordiv: operations.floordiv,
    operator.mod: operations.mod,
    operator.pow: operations.pow,
    operator.eq: operations.eq,
    operator.ne: operations.ne,
    operator.lt: operations.lt,
    operator.gt: operations.gt,
    operator.le: operations.le,
    operator.ge: operations.ge,
    operator.pos: operations.pos,
    operator.neg: operations.neg,
    operator.not_: operations.not_,
    operator.and_: operations.and_,
    operator.or_: operations.or_,
    operator.getitem: operations.getitem,
    math.floor: P.scalar_floor,
    math.trunc: P.scalar_trunc,
    math.exp: operations.scalar_exp,
    math.log: operations.scalar_log,
    math.sin: operations.scalar_sin,
    math.cos: operations.scalar_cos,
    math.tan: operations.scalar_tan,
    math.tanh: operations.scalar_tanh,
    np.floor: operations.array_floor,
    np.trunc: operations.array_trunc,
    np.add: operations.array_add,
    np.subtract: operations.array_sub,
    np.multiply: operations.array_mul,
    np.divide: operations.array_truediv,
    np.mod: operations.array_mod,
    np.power: operations.array_pow,
    np.exp: operations.array_exp,
    np.log: operations.array_log,
    np.sin: operations.array_sin,
    np.cos: operations.array_cos,
    np.tan: operations.array_tan,
    np.tanh: operations.array_tanh,
    np.sum: operations.sum,
}


scalar_object_map = {
    **python_operation_map,
    operations.add: P.scalar_add,
    operations.sub: P.scalar_sub,
    operations.mul: P.scalar_mul,
    operations.mod: P.scalar_mod,
    operations.pow: P.scalar_pow,
    operations.eq: P.scalar_eq,
    operations.ne: P.scalar_ne,
    operations.lt: P.scalar_lt,
    operations.gt: P.scalar_gt,
    operations.le: P.scalar_le,
    operations.ge: P.scalar_ge,
    operations.pos: P.scalar_uadd,
    operations.neg: P.scalar_usub,
    operations.not_: P.bool_not,
    operations.and_: P.bool_and,
    operations.or_: P.bool_or,
    operations.getitem: P.tuple_getitem,
    operations.bool: P.identity,
    operations.user_switch: P.switch,
}


standard_object_map = {
    **python_operation_map,
}


standard_method_map = {
    xtype.Nil: {
        '__eq__': operations.nil_eq,
        '__ne__': operations.nil_ne,
        '__bool__': operations.nil_bool,
    },
    xtype.Bool: {
        '__and__': operations.bool_and,
        '__or__': operations.bool_or,
        '__eq__': operations.bool_eq,
        '__ne__': operations.bool_ne,
        '__bool__': operations.identity,
    },
    xtype.String: {
        '__eq__': operations.string_eq,
        '__ne__': operations.string_ne,
    },
    xtype.Number: {
        '__add__': lop(operations.scalar_add, xtype.Number, '__add__'),
        '__sub__': lop(operations.scalar_sub, xtype.Number, '__sub__'),
        '__mul__': lop(operations.scalar_mul, xtype.Number, '__mul__'),
        '__mod__': lop(operations.scalar_mod, xtype.Number, '__mod__'),
        '__pow__': lop(operations.scalar_pow, xtype.Number, '__pow__'),
        '__radd__': rop(operations.scalar_add, xtype.Number, '__radd__'),
        '__rsub__': rop(operations.scalar_sub, xtype.Number, '__rsub__'),
        '__rmul__': rop(operations.scalar_mul, xtype.Number, '__rmul__'),
        '__rmod__': rop(operations.scalar_mod, xtype.Number, '__rmod__'),
        '__rpow__': rop(operations.scalar_pow, xtype.Number, '__rpow__'),
        '__pos__': operations.scalar_uadd,
        '__neg__': operations.scalar_usub,
        '__eq__': lop(operations.scalar_eq, xtype.Number, '__eq__'),
        '__ne__': lop(operations.scalar_ne, xtype.Number, '__ne__'),
        '__lt__': lop(operations.scalar_lt, xtype.Number, '__lt__'),
        '__gt__': lop(operations.scalar_gt, xtype.Number, '__gt__'),
        '__le__': lop(operations.scalar_le, xtype.Number, '__le__'),
        '__ge__': lop(operations.scalar_ge, xtype.Number, '__ge__'),
    },
    xtype.Int: {
        '__floordiv__': lop(operations.int_floordiv, xtype.Int,
                            '__floordiv__'),
        '__truediv__': lop(operations.int_truediv, xtype.Int,
                           '__truediv__'),
        '__rfloordiv__': rop(operations.int_floordiv, xtype.Int,
                             '__rfloordiv__'),
        '__rtruediv__': rop(operations.int_truediv, xtype.Int,
                            '__rtruediv__'),
        '__floor__': operations.identity,
        '__trunc__': operations.identity,
        '__bool__': operations.int_bool,
        '__myia_to_array__': operations.scalar_to_array,
    },
    xtype.UInt: {
        '__floordiv__': lop(operations.int_floordiv, xtype.UInt,
                            '__floordiv__'),
        '__truediv__': lop(operations.int_truediv, xtype.UInt,
                           '__truediv__'),
        '__rfloordiv__': rop(operations.int_floordiv, xtype.UInt,
                             '__rfloordiv__'),
        '__rtruediv__': rop(operations.int_truediv, xtype.UInt,
                            '__rtruediv__'),
        '__floor__': operations.identity,
        '__trunc__': operations.identity,
        '__bool__': operations.int_bool,
        '__myia_to_array__': operations.scalar_to_array,
    },
    xtype.Float: {
        '__floordiv__': lop(operations.float_floordiv, xtype.Float,
                            '__floordiv__'),
        '__truediv__': lop(operations.scalar_div, xtype.Float,
                           '__truediv__'),
        '__rfloordiv__': rop(operations.float_floordiv, xtype.Float,
                             '__rfloordiv__'),
        '__rtruediv__': rop(operations.scalar_div, xtype.Float,
                            '__rtruediv__'),
        '__floor__': operations.scalar_floor,
        '__trunc__': operations.scalar_trunc,
        '__bool__': operations.float_bool,
        '__myia_to_array__': operations.scalar_to_array,
    },
    xtype.Tuple: {
        '__len__': operations.tuple_len,
        '__add__': operations.tuple_concat,
        '__getitem__': operations.tuple_get,
        '__setitem__': operations.tuple_setitem,
        '__myia_iter__': operations.identity,
        '__myia_next__': operations.tuple_next,
        '__myia_hasnext__': operations.tuple_hasnext,
    },
    xtype.Dict: {
        '__getitem__': operations.dict_getitem,
        'values': operations.dict_values,
    },
    xtype.NDArray: {
        '__add__': operations.array_add,
        '__sub__': operations.array_sub,
        '__mul__': operations.array_mul,
        '__truediv__': operations.array_truediv,
        '__floordiv__': operations.array_floordiv,
        '__mod__': operations.array_mod,
        '__pow__': operations.array_pow,
        '__floor__': operations.array_floor,
        '__trunc__': operations.array_trunc,
        '__radd__': reverse_binop(operations.array_add, '__radd__'),
        '__rsub__': reverse_binop(operations.array_sub, '__rsub__'),
        '__rmul__': reverse_binop(operations.array_mul, '__rmul__'),
        '__rtruediv__': reverse_binop(operations.array_truediv,
                                      '__rtruediv__'),
        '__rfloordiv__': reverse_binop(operations.array_floordiv,
                                       '__rfloordiv__'),
        '__rmod__': reverse_binop(operations.array_mod, '__rmod__'),
        '__rpow__': reverse_binop(operations.array_pow, '__rpow__'),
        '__pos__': operations.array_uadd,
        '__neg__': operations.array_usub,
        '__eq__': operations.array_eq,
        '__ne__': operations.array_ne,
        '__lt__': operations.array_lt,
        '__gt__': operations.array_gt,
        '__le__': operations.array_le,
        '__ge__': operations.array_ge,
        '__matmul__': operations.dot,
        '__len__': operations.array_len,
        '__getitem__': operations.array_getitem_wrap,
        '__myia_iter__': operations.array_iter,
        '__myia_to_array__': operations.identity,
        'item': operations.array_to_scalar,
        'shape': property(operations.shape),
        'T': property(operations.t),
        'ndim': property(operations.ndim),
        'dtype': property(operations.dtype),
    },
    object: {
        '__eq__': ABSENT,
        '__ne__': ABSENT,
        '__lt__': ABSENT,
        '__gt__': ABSENT,
        '__le__': ABSENT,
        '__ge__': ABSENT,
    }
}


standard_resources = Resources.partial(
    manager=GraphManager.partial(),
    py_implementations=py_registry,
    grad_implementations=grad_registry,
    method_map=standard_method_map,
    convert=ConverterResource.partial(
        object_map=standard_object_map,
    ),
    inferrer=InferenceResource.partial(
        constructors=inferrer_registry,
        context_class=Context,
    ),
    backend=BackendResource.partial(),
    debug_vm=DebugVMResource.partial(
        implementations=vm_registry,
    ),
    operation_whitelist=default_whitelist,
    validate_abstract=default_validate_abstract,
    return_backend=False,
)


######################
# Pre-made pipelines #
######################


standard_pipeline = PipelineDefinition(
    resources=standard_resources,
    parse=steps.step_parse,
    resolve=steps.step_resolve,
    infer=steps.step_infer,
    specialize=steps.step_specialize,
    simplify_types=steps.step_simplify_types,
    opt=steps.step_opt,
    opt2=steps.step_opt2,
    validate=steps.step_validate,
    compile=steps.step_compile,
    wrap=steps.step_wrap,
)


scalar_pipeline = standard_pipeline.configure({
    'resources.convert.object_map': scalar_object_map,
})


standard_debug_pipeline = PipelineDefinition(
    resources=standard_resources,
    parse=steps.step_parse,
    resolve=steps.step_resolve,
    infer=steps.step_infer,
    specialize=steps.step_specialize,
    simplify_types=steps.step_simplify_types,
    opt=steps.step_opt,
    opt2=steps.step_opt2,
    validate=steps.step_validate,
    export=steps.step_debug_export,
    wrap=steps.step_wrap,
).configure({
    'resources.backend.name': False
})


scalar_debug_pipeline = standard_debug_pipeline.configure({
    'resources.convert.object_map': scalar_object_map
})


######################
# Pre-made utilities #
######################


standard_parse = standard_pipeline \
    .select('resources', 'parse') \
    .make_transformer('input', 'graph')


scalar_parse = scalar_pipeline \
    .select('resources', 'parse', 'resolve') \
    .make_transformer('input', 'graph')


scalar_debug_compile = scalar_debug_pipeline \
    .select('resources', 'parse', 'resolve', 'export') \
    .make_transformer('input', 'output')


__consolidate__ = True
__all__ = [
    'py_registry',
    'scalar_debug_compile',
    'scalar_debug_pipeline',
    'scalar_parse',
    'scalar_pipeline',
    'standard_debug_pipeline',
    'standard_parse',
    'standard_pipeline',
    'standard_resources',
    'vm_registry',
]
