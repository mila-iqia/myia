"""Primitive operations.

Primitive operations are handled as constants in the intermediate
representation, with the constant's value being an instance of a `Primitive`
subclass.

"""

###############################################################################
# THIS FILE IS GENERATED AUTOMATICALLY. DO NOT EDIT!                          #
# To regenerate this file, run `python scripts/regen.py`                      #
# The script will search for all primitives it can find in myia.operations    #
###############################################################################

from .utils import BackendPrimitive, InferencePrimitive, PlaceholderPrimitive

J = PlaceholderPrimitive(
    name='J',
    defaults='myia.operations.prim_J'
)

Jinv = PlaceholderPrimitive(
    name='Jinv',
    defaults='myia.operations.prim_Jinv'
)

argmax = BackendPrimitive(
    name='argmax',
    defaults='myia.operations.prim_argmax'
)

array_cast = BackendPrimitive(
    name='array_cast',
    defaults='myia.operations.prim_array_cast'
)

array_getitem = BackendPrimitive(
    name='array_getitem',
    defaults='myia.operations.prim_array_getitem'
)

array_map = BackendPrimitive(
    name='array_map',
    defaults='myia.operations.prim_array_map'
)

array_max = BackendPrimitive(
    name='array_max',
    defaults='myia.operations.prim_array_max'
)

array_reduce = BackendPrimitive(
    name='array_reduce',
    defaults='myia.operations.prim_array_reduce'
)

array_scan = BackendPrimitive(
    name='array_scan',
    defaults='myia.operations.prim_array_scan'
)

array_setitem = BackendPrimitive(
    name='array_setitem',
    defaults='myia.operations.prim_array_setitem'
)

array_to_scalar = BackendPrimitive(
    name='array_to_scalar',
    defaults='myia.operations.prim_array_to_scalar'
)

bool_and = BackendPrimitive(
    name='bool_and',
    defaults='myia.operations.prim_bool_and'
)

bool_eq = BackendPrimitive(
    name='bool_eq',
    defaults='myia.operations.prim_bool_eq'
)

bool_not = BackendPrimitive(
    name='bool_not',
    defaults='myia.operations.prim_bool_not'
)

bool_or = BackendPrimitive(
    name='bool_or',
    defaults='myia.operations.prim_bool_or'
)

broadcast_shape = BackendPrimitive(
    name='broadcast_shape',
    defaults='myia.operations.prim_broadcast_shape'
)

casttag = BackendPrimitive(
    name='casttag',
    defaults='myia.operations.prim_casttag'
)

concat = BackendPrimitive(
    name='concat',
    defaults='myia.operations.prim_concat'
)

conv2d = BackendPrimitive(
    name='conv2d',
    defaults='myia.operations.prim_conv2d'
)

conv2d_input_grad = BackendPrimitive(
    name='conv2d_input_grad',
    defaults='myia.operations.prim_conv2d_input_grad'
)

conv2d_weight_grad = BackendPrimitive(
    name='conv2d_weight_grad',
    defaults='myia.operations.prim_conv2d_weight_grad'
)

dict_getitem = InferencePrimitive(
    name='dict_getitem',
    defaults='myia.operations.prim_dict_getitem'
)

dict_setitem = InferencePrimitive(
    name='dict_setitem',
    defaults='myia.operations.prim_dict_setitem'
)

distribute = BackendPrimitive(
    name='distribute',
    defaults='myia.operations.prim_distribute'
)

dot = BackendPrimitive(
    name='dot',
    defaults='myia.operations.prim_dot'
)

embedding = BackendPrimitive(
    name='embedding',
    defaults='myia.operations.prim_embedding'
)

env_add = BackendPrimitive(
    name='env_add',
    defaults='myia.operations.prim_env_add'
)

env_getitem = BackendPrimitive(
    name='env_getitem',
    defaults='myia.operations.prim_env_getitem'
)

env_setitem = BackendPrimitive(
    name='env_setitem',
    defaults='myia.operations.prim_env_setitem'
)

extract_kwarg = InferencePrimitive(
    name='extract_kwarg',
    defaults='myia.operations.prim_extract_kwarg'
)

gather = BackendPrimitive(
    name='gather',
    defaults='myia.operations.prim_gather'
)

grad_embedding_weights = BackendPrimitive(
    name='grad_embedding_weights',
    defaults='myia.operations.prim_grad_embedding_weights'
)

handle = BackendPrimitive(
    name='handle',
    defaults='myia.operations.prim_handle'
)

hastag = BackendPrimitive(
    name='hastag',
    defaults='myia.operations.prim_hastag'
)

hastype = InferencePrimitive(
    name='hastype',
    defaults='myia.operations.prim_hastype'
)

identity = PlaceholderPrimitive(
    name='identity',
    defaults='myia.operations.prim_identity'
)

invert_permutation = BackendPrimitive(
    name='invert_permutation',
    defaults='myia.operations.prim_invert_permutation'
)

make_dict = InferencePrimitive(
    name='make_dict',
    defaults='myia.operations.prim_make_dict'
)

make_exception = BackendPrimitive(
    name='make_exception',
    defaults='myia.operations.prim_make_exception'
)

make_kwarg = InferencePrimitive(
    name='make_kwarg',
    defaults='myia.operations.prim_make_kwarg'
)

make_record = InferencePrimitive(
    name='make_record',
    defaults='myia.operations.prim_make_record'
)

make_tuple = BackendPrimitive(
    name='make_tuple',
    defaults='myia.operations.prim_make_tuple'
)

max_pool2d = BackendPrimitive(
    name='max_pool2d',
    defaults='myia.operations.prim_max_pool2d'
)

max_pool2d_grad = BackendPrimitive(
    name='max_pool2d_grad',
    defaults='myia.operations.prim_max_pool2d_grad'
)

partial = BackendPrimitive(
    name='partial',
    defaults='myia.operations.prim_partial'
)

raise_ = BackendPrimitive(
    name='raise',
    defaults='myia.operations.prim_raise_'
)

record_getitem = InferencePrimitive(
    name='record_getitem',
    defaults='myia.operations.prim_record_getitem'
)

record_setitem = InferencePrimitive(
    name='record_setitem',
    defaults='myia.operations.prim_record_setitem'
)

reshape = BackendPrimitive(
    name='reshape',
    defaults='myia.operations.prim_reshape'
)

return_ = BackendPrimitive(
    name='return',
    defaults='myia.operations.prim_return_'
)

scalar_abs = BackendPrimitive(
    name='scalar_abs',
    defaults='myia.operations.prim_scalar_abs'
)

scalar_add = BackendPrimitive(
    name='scalar_add',
    defaults='myia.operations.prim_scalar_add'
)

scalar_bit_and = BackendPrimitive(
    name='scalar_bit_and',
    defaults='myia.operations.prim_scalar_bit_and'
)

scalar_bit_lshift = BackendPrimitive(
    name='scalar_bit_lshift',
    defaults='myia.operations.prim_scalar_bit_lshift'
)

scalar_bit_or = BackendPrimitive(
    name='scalar_bit_or',
    defaults='myia.operations.prim_scalar_bit_or'
)

scalar_bit_rshift = BackendPrimitive(
    name='scalar_bit_rshift',
    defaults='myia.operations.prim_scalar_bit_rshift'
)

scalar_bit_xor = BackendPrimitive(
    name='scalar_bit_xor',
    defaults='myia.operations.prim_scalar_bit_xor'
)

scalar_cast = BackendPrimitive(
    name='scalar_cast',
    defaults='myia.operations.prim_scalar_cast'
)

scalar_cos = BackendPrimitive(
    name='scalar_cos',
    defaults='myia.operations.prim_scalar_cos'
)

scalar_div = BackendPrimitive(
    name='scalar_div',
    defaults='myia.operations.prim_scalar_div'
)

scalar_eq = BackendPrimitive(
    name='scalar_eq',
    defaults='myia.operations.prim_scalar_eq'
)

scalar_exp = BackendPrimitive(
    name='scalar_exp',
    defaults='myia.operations.prim_scalar_exp'
)

scalar_floor = BackendPrimitive(
    name='scalar_floor',
    defaults='myia.operations.prim_scalar_floor'
)

scalar_ge = BackendPrimitive(
    name='scalar_ge',
    defaults='myia.operations.prim_scalar_ge'
)

scalar_gt = BackendPrimitive(
    name='scalar_gt',
    defaults='myia.operations.prim_scalar_gt'
)

scalar_le = BackendPrimitive(
    name='scalar_le',
    defaults='myia.operations.prim_scalar_le'
)

scalar_log = BackendPrimitive(
    name='scalar_log',
    defaults='myia.operations.prim_scalar_log'
)

scalar_lt = BackendPrimitive(
    name='scalar_lt',
    defaults='myia.operations.prim_scalar_lt'
)

scalar_max = BackendPrimitive(
    name='scalar_max',
    defaults='myia.operations.prim_scalar_max'
)

scalar_mod = BackendPrimitive(
    name='scalar_mod',
    defaults='myia.operations.prim_scalar_mod'
)

scalar_mul = BackendPrimitive(
    name='scalar_mul',
    defaults='myia.operations.prim_scalar_mul'
)

scalar_ne = BackendPrimitive(
    name='scalar_ne',
    defaults='myia.operations.prim_scalar_ne'
)

scalar_pow = BackendPrimitive(
    name='scalar_pow',
    defaults='myia.operations.prim_scalar_pow'
)

scalar_sign = BackendPrimitive(
    name='scalar_sign',
    defaults='myia.operations.prim_scalar_sign'
)

scalar_sin = BackendPrimitive(
    name='scalar_sin',
    defaults='myia.operations.prim_scalar_sin'
)

scalar_sub = BackendPrimitive(
    name='scalar_sub',
    defaults='myia.operations.prim_scalar_sub'
)

scalar_tan = BackendPrimitive(
    name='scalar_tan',
    defaults='myia.operations.prim_scalar_tan'
)

scalar_tanh = BackendPrimitive(
    name='scalar_tanh',
    defaults='myia.operations.prim_scalar_tanh'
)

scalar_to_array = BackendPrimitive(
    name='scalar_to_array',
    defaults='myia.operations.prim_scalar_to_array'
)

scalar_trunc = BackendPrimitive(
    name='scalar_trunc',
    defaults='myia.operations.prim_scalar_trunc'
)

scalar_uadd = BackendPrimitive(
    name='scalar_uadd',
    defaults='myia.operations.prim_scalar_uadd'
)

scalar_usub = BackendPrimitive(
    name='scalar_usub',
    defaults='myia.operations.prim_scalar_usub'
)

scatter = BackendPrimitive(
    name='scatter',
    defaults='myia.operations.prim_scatter'
)

scatter_add = BackendPrimitive(
    name='scatter_add',
    defaults='myia.operations.prim_scatter_add'
)

shape = BackendPrimitive(
    name='shape',
    defaults='myia.operations.prim_shape'
)

split = BackendPrimitive(
    name='split',
    defaults='myia.operations.prim_split'
)

stop_gradient = PlaceholderPrimitive(
    name='stop_gradient',
    defaults='myia.operations.prim_stop_gradient'
)

string_eq = InferencePrimitive(
    name='string_eq',
    defaults='myia.operations.prim_string_eq'
)

switch = BackendPrimitive(
    name='switch',
    defaults='myia.operations.prim_switch'
)

tagged = BackendPrimitive(
    name='tagged',
    defaults='myia.operations.prim_tagged'
)

transpose = BackendPrimitive(
    name='transpose',
    defaults='myia.operations.prim_transpose'
)

tuple_getitem = BackendPrimitive(
    name='tuple_getitem',
    defaults='myia.operations.prim_tuple_getitem'
)

tuple_setitem = BackendPrimitive(
    name='tuple_setitem',
    defaults='myia.operations.prim_tuple_setitem'
)

universe_getitem = BackendPrimitive(
    name='universe_getitem',
    defaults='myia.operations.prim_universe_getitem'
)

universe_setitem = BackendPrimitive(
    name='universe_setitem',
    defaults='myia.operations.prim_universe_setitem'
)

unsafe_static_cast = BackendPrimitive(
    name='unsafe_static_cast',
    defaults='myia.operations.prim_unsafe_static_cast'
)
