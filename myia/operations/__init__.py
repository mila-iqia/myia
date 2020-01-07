"""Myia operations."""

###############################################################################
# THIS FILE IS GENERATED AUTOMATICALLY. DO NOT EDIT!                          #
# To regenerate this file, run `python scripts/regen.py`                      #
# The script will search for all operations it can find in myia.operations    #
###############################################################################

from .utils import Operation, Primitive  # noqa

J = Operation(
    name='J',
    defaults='myia.operations.prim_J'
)

Jinv = Operation(
    name='Jinv',
    defaults='myia.operations.prim_Jinv'
)

abstract_array = Operation(
    name='abstract_array',
    defaults='myia.operations.macro_abstract_array'
)

add = Operation(
    name='add',
    defaults='myia.operations.ops_dunder.add'
)

and_ = Operation(
    name='and',
    defaults='myia.operations.ops_dunder.and_'
)

apply = Operation(
    name='apply',
    defaults='myia.operations.macro_apply'
)

argmax = Operation(
    name='argmax',
    defaults='myia.operations.prim_argmax'
)

array_abs = Operation(
    name='array_abs',
    defaults='myia.operations.ops_array.array_abs'
)

array_add = Operation(
    name='array_add',
    defaults='myia.operations.ops_array.array_add'
)

array_cast = Operation(
    name='array_cast',
    defaults='myia.operations.prim_array_cast'
)

array_cos = Operation(
    name='array_cos',
    defaults='myia.operations.ops_array.array_cos'
)

array_eq = Operation(
    name='array_eq',
    defaults='myia.operations.ops_array.array_eq'
)

array_exp = Operation(
    name='array_exp',
    defaults='myia.operations.ops_array.array_exp'
)

array_floor = Operation(
    name='array_floor',
    defaults='myia.operations.ops_array.array_floor'
)

array_floordiv = Operation(
    name='array_floordiv',
    defaults='myia.operations.ops_array.array_floordiv'
)

array_ge = Operation(
    name='array_ge',
    defaults='myia.operations.ops_array.array_ge'
)

array_getitem = Operation(
    name='array_getitem',
    defaults='myia.operations.prim_array_getitem'
)

array_getitem_wrap = Operation(
    name='array_getitem_wrap',
    defaults='myia.operations.op_array_getitem_wrap'
)

array_gt = Operation(
    name='array_gt',
    defaults='myia.operations.ops_array.array_gt'
)

array_iter = Operation(
    name='array_iter',
    defaults='myia.operations.ops_array.array_iter'
)

array_le = Operation(
    name='array_le',
    defaults='myia.operations.ops_array.array_le'
)

array_len = Operation(
    name='array_len',
    defaults='myia.operations.macro_array_len'
)

array_log = Operation(
    name='array_log',
    defaults='myia.operations.ops_array.array_log'
)

array_lt = Operation(
    name='array_lt',
    defaults='myia.operations.ops_array.array_lt'
)

array_map = Operation(
    name='array_map',
    defaults='myia.operations.prim_array_map'
)

array_max = Operation(
    name='array_max',
    defaults='myia.operations.prim_array_max'
)

array_mod = Operation(
    name='array_mod',
    defaults='myia.operations.ops_array.array_mod'
)

array_mul = Operation(
    name='array_mul',
    defaults='myia.operations.ops_array.array_mul'
)

array_ne = Operation(
    name='array_ne',
    defaults='myia.operations.ops_array.array_ne'
)

array_pow = Operation(
    name='array_pow',
    defaults='myia.operations.ops_array.array_pow'
)

array_reduce = Operation(
    name='array_reduce',
    defaults='myia.operations.prim_array_reduce'
)

array_reduce_dim = Operation(
    name='array_reduce_dim',
    defaults='myia.operations.op_array_reduce_dim'
)

array_scan = Operation(
    name='array_scan',
    defaults='myia.operations.prim_array_scan'
)

array_setitem = Operation(
    name='array_setitem',
    defaults='myia.operations.prim_array_setitem'
)

array_sign = Operation(
    name='array_sign',
    defaults='myia.operations.ops_array.array_sign'
)

array_sin = Operation(
    name='array_sin',
    defaults='myia.operations.ops_array.array_sin'
)

array_sub = Operation(
    name='array_sub',
    defaults='myia.operations.ops_array.array_sub'
)

array_tan = Operation(
    name='array_tan',
    defaults='myia.operations.ops_array.array_tan'
)

array_tanh = Operation(
    name='array_tanh',
    defaults='myia.operations.ops_array.array_tanh'
)

array_to_scalar = Operation(
    name='array_to_scalar',
    defaults='myia.operations.prim_array_to_scalar'
)

array_truediv = Operation(
    name='array_truediv',
    defaults='myia.operations.ops_array.array_truediv'
)

array_trunc = Operation(
    name='array_trunc',
    defaults='myia.operations.ops_array.array_trunc'
)

array_uadd = Operation(
    name='array_uadd',
    defaults='myia.operations.ops_array.array_uadd'
)

array_usub = Operation(
    name='array_usub',
    defaults='myia.operations.ops_array.array_usub'
)

bool = Operation(
    name='bool',
    defaults='myia.operations.ops_dunder.bool'
)

bool_and = Operation(
    name='bool_and',
    defaults='myia.operations.prim_bool_and'
)

bool_eq = Operation(
    name='bool_eq',
    defaults='myia.operations.prim_bool_eq'
)

bool_ne = Operation(
    name='bool_ne',
    defaults='myia.operations.ops_bool.bool_ne'
)

bool_not = Operation(
    name='bool_not',
    defaults='myia.operations.prim_bool_not'
)

bool_or = Operation(
    name='bool_or',
    defaults='myia.operations.prim_bool_or'
)

broadcast_shape = Operation(
    name='broadcast_shape',
    defaults='myia.operations.prim_broadcast_shape'
)

call_object = Operation(
    name='call_object',
    defaults='myia.operations.macro_call_object'
)

casttag = Operation(
    name='casttag',
    defaults='myia.operations.prim_casttag'
)

concat = Operation(
    name='concat',
    defaults='myia.operations.prim_concat'
)

conv2d = Operation(
    name='conv2d',
    defaults='myia.operations.prim_conv2d'
)

conv2d_input_grad = Operation(
    name='conv2d_input_grad',
    defaults='myia.operations.prim_conv2d_input_grad'
)

conv2d_weight_grad = Operation(
    name='conv2d_weight_grad',
    defaults='myia.operations.prim_conv2d_weight_grad'
)

conv_transpose2d = Operation(
    name='conv_transpose2d',
    defaults='myia.operations.prim_conv_transpose2d'
)

dict_getitem = Operation(
    name='dict_getitem',
    defaults='myia.operations.prim_dict_getitem'
)

dict_setitem = Operation(
    name='dict_setitem',
    defaults='myia.operations.prim_dict_setitem'
)

dict_values = Operation(
    name='dict_values',
    defaults='myia.operations.macro_dict_values'
)

distribute = Operation(
    name='distribute',
    defaults='myia.operations.prim_distribute'
)

dot = Operation(
    name='dot',
    defaults='myia.operations.prim_dot'
)

dtype = Operation(
    name='dtype',
    defaults='myia.operations.macro_dtype'
)

embed = Operation(
    name='embed',
    defaults='myia.operations.macro_embed'
)

enumerate = Operation(
    name='enumerate',
    defaults='myia.operations.op_enumerate'
)

env_add = Operation(
    name='env_add',
    defaults='myia.operations.prim_env_add'
)

env_getitem = Operation(
    name='env_getitem',
    defaults='myia.operations.prim_env_getitem'
)

env_setitem = Operation(
    name='env_setitem',
    defaults='myia.operations.prim_env_setitem'
)

eq = Operation(
    name='eq',
    defaults='myia.operations.ops_dunder.eq'
)

extract_kwarg = Operation(
    name='extract_kwarg',
    defaults='myia.operations.prim_extract_kwarg'
)

float_bool = Operation(
    name='float_bool',
    defaults='myia.operations.ops_scalar.float_bool'
)

float_floordiv = Operation(
    name='float_floordiv',
    defaults='myia.operations.ops_scalar.float_floordiv'
)

floor = Operation(
    name='floor',
    defaults='myia.operations.ops_dunder.floor'
)

floordiv = Operation(
    name='floordiv',
    defaults='myia.operations.ops_dunder.floordiv'
)

full = Operation(
    name='full',
    defaults='myia.operations.op_full'
)

gadd = Operation(
    name='gadd',
    defaults='myia.operations.op_gadd'
)

gather = Operation(
    name='gather',
    defaults='myia.operations.prim_gather'
)

ge = Operation(
    name='ge',
    defaults='myia.operations.ops_dunder.ge'
)

getattr = Operation(
    name='getattr',
    defaults='myia.operations.macro_getattr'
)

getitem = Operation(
    name='getitem',
    defaults='myia.operations.ops_dunder.getitem'
)

grad = Operation(
    name='grad',
    defaults='myia.operations.macro_grad'
)

gt = Operation(
    name='gt',
    defaults='myia.operations.ops_dunder.gt'
)

handle = Operation(
    name='handle',
    defaults='myia.operations.prim_handle'
)

handle_get = Operation(
    name='handle_get',
    defaults='myia.operations.ops_universe.handle_get'
)

handle_set = Operation(
    name='handle_set',
    defaults='myia.operations.ops_universe.handle_set'
)

hasattr = Operation(
    name='hasattr',
    defaults='myia.operations.macro_hasattr'
)

hastag = Operation(
    name='hastag',
    defaults='myia.operations.prim_hastag'
)

hastype = Operation(
    name='hastype',
    defaults='myia.operations.prim_hastype'
)

hyper_map = Operation(
    name='hyper_map',
    defaults='myia.operations.op_hyper_map'
)

identity = Operation(
    name='identity',
    defaults='myia.operations.prim_identity'
)

int_bool = Operation(
    name='int_bool',
    defaults='myia.operations.ops_scalar.int_bool'
)

int_floordiv = Operation(
    name='int_floordiv',
    defaults='myia.operations.ops_scalar.int_floordiv'
)

int_truediv = Operation(
    name='int_truediv',
    defaults='myia.operations.ops_scalar.int_truediv'
)

invert_permutation = Operation(
    name='invert_permutation',
    defaults='myia.operations.prim_invert_permutation'
)

is_ = Operation(
    name='is',
    defaults='myia.operations.macro_is'
)

is_not = Operation(
    name='is_not',
    defaults='myia.operations.op_is_not'
)

isinstance = Operation(
    name='isinstance',
    defaults='myia.operations.macro_isinstance'
)

le = Operation(
    name='le',
    defaults='myia.operations.ops_dunder.le'
)

len = Operation(
    name='len',
    defaults='myia.operations.ops_dunder.len'
)

lshift = Operation(
    name='lshift',
    defaults='myia.operations.ops_dunder.lshift'
)

lt = Operation(
    name='lt',
    defaults='myia.operations.ops_dunder.lt'
)

make_dict = Operation(
    name='make_dict',
    defaults='myia.operations.prim_make_dict'
)

make_exception = Operation(
    name='make_exception',
    defaults='myia.operations.prim_make_exception'
)

make_kwarg = Operation(
    name='make_kwarg',
    defaults='myia.operations.prim_make_kwarg'
)

make_list = Operation(
    name='make_list',
    defaults='myia.operations.macro_make_list'
)

make_record = Operation(
    name='make_record',
    defaults='myia.operations.prim_make_record'
)

make_tuple = Operation(
    name='make_tuple',
    defaults='myia.operations.prim_make_tuple'
)

matmul = Operation(
    name='matmul',
    defaults='myia.operations.ops_dunder.matmul'
)

max_pool2d = Operation(
    name='max_pool2d',
    defaults='myia.operations.prim_max_pool2d'
)

max_pool2d_grad = Operation(
    name='max_pool2d_grad',
    defaults='myia.operations.prim_max_pool2d_grad'
)

mod = Operation(
    name='mod',
    defaults='myia.operations.ops_dunder.mod'
)

mul = Operation(
    name='mul',
    defaults='myia.operations.ops_dunder.mul'
)

myia_hasnext = Operation(
    name='myia_hasnext',
    defaults='myia.operations.ops_dunder.myia_hasnext'
)

myia_iter = Operation(
    name='myia_iter',
    defaults='myia.operations.ops_dunder.myia_iter'
)

myia_next = Operation(
    name='myia_next',
    defaults='myia.operations.ops_dunder.myia_next'
)

myia_to_array = Operation(
    name='myia_to_array',
    defaults='myia.operations.ops_dunder.myia_to_array'
)

ndim = Operation(
    name='ndim',
    defaults='myia.operations.ops_array.ndim'
)

ne = Operation(
    name='ne',
    defaults='myia.operations.ops_dunder.ne'
)

neg = Operation(
    name='neg',
    defaults='myia.operations.ops_dunder.neg'
)

nil_bool = Operation(
    name='nil_bool',
    defaults='myia.operations.ops_nil.nil_bool'
)

nil_eq = Operation(
    name='nil_eq',
    defaults='myia.operations.ops_nil.nil_eq'
)

nil_ne = Operation(
    name='nil_ne',
    defaults='myia.operations.ops_nil.nil_ne'
)

not_ = Operation(
    name='not_',
    defaults='myia.operations.ops_bool.not_'
)

or_ = Operation(
    name='or',
    defaults='myia.operations.ops_dunder.or_'
)

partial = Operation(
    name='partial',
    defaults='myia.operations.prim_partial'
)

pos = Operation(
    name='pos',
    defaults='myia.operations.ops_dunder.pos'
)

pow = Operation(
    name='pow',
    defaults='myia.operations.ops_dunder.pow'
)

prod = Operation(
    name='prod',
    defaults='myia.operations.ops_array.prod'
)

raise_ = Operation(
    name='raise',
    defaults='myia.operations.prim_raise_'
)

range = Operation(
    name='range',
    defaults='myia.operations.op_range'
)

record_getitem = Operation(
    name='record_getitem',
    defaults='myia.operations.prim_record_getitem'
)

record_setitem = Operation(
    name='record_setitem',
    defaults='myia.operations.prim_record_setitem'
)

reshape = Operation(
    name='reshape',
    defaults='myia.operations.prim_reshape'
)

resolve = Operation(
    name='resolve',
    defaults='myia.operations.macro_resolve'
)

return_ = Operation(
    name='return',
    defaults='myia.operations.prim_return_'
)

rshift = Operation(
    name='rshift',
    defaults='myia.operations.ops_dunder.rshift'
)

scalar_abs = Operation(
    name='scalar_abs',
    defaults='myia.operations.prim_scalar_abs'
)

scalar_add = Operation(
    name='scalar_add',
    defaults='myia.operations.prim_scalar_add'
)

scalar_bit_and = Operation(
    name='scalar_bit_and',
    defaults='myia.operations.prim_scalar_bit_and'
)

scalar_bit_lshift = Operation(
    name='scalar_bit_lshift',
    defaults='myia.operations.prim_scalar_bit_lshift'
)

scalar_bit_or = Operation(
    name='scalar_bit_or',
    defaults='myia.operations.prim_scalar_bit_or'
)

scalar_bit_rshift = Operation(
    name='scalar_bit_rshift',
    defaults='myia.operations.prim_scalar_bit_rshift'
)

scalar_bit_xor = Operation(
    name='scalar_bit_xor',
    defaults='myia.operations.prim_scalar_bit_xor'
)

scalar_cast = Operation(
    name='scalar_cast',
    defaults='myia.operations.prim_scalar_cast'
)

scalar_cos = Operation(
    name='scalar_cos',
    defaults='myia.operations.prim_scalar_cos'
)

scalar_div = Operation(
    name='scalar_div',
    defaults='myia.operations.prim_scalar_div'
)

scalar_eq = Operation(
    name='scalar_eq',
    defaults='myia.operations.prim_scalar_eq'
)

scalar_exp = Operation(
    name='scalar_exp',
    defaults='myia.operations.prim_scalar_exp'
)

scalar_floor = Operation(
    name='scalar_floor',
    defaults='myia.operations.prim_scalar_floor'
)

scalar_ge = Operation(
    name='scalar_ge',
    defaults='myia.operations.prim_scalar_ge'
)

scalar_gt = Operation(
    name='scalar_gt',
    defaults='myia.operations.prim_scalar_gt'
)

scalar_le = Operation(
    name='scalar_le',
    defaults='myia.operations.prim_scalar_le'
)

scalar_log = Operation(
    name='scalar_log',
    defaults='myia.operations.prim_scalar_log'
)

scalar_lt = Operation(
    name='scalar_lt',
    defaults='myia.operations.prim_scalar_lt'
)

scalar_max = Operation(
    name='scalar_max',
    defaults='myia.operations.prim_scalar_max'
)

scalar_mod = Operation(
    name='scalar_mod',
    defaults='myia.operations.prim_scalar_mod'
)

scalar_mul = Operation(
    name='scalar_mul',
    defaults='myia.operations.prim_scalar_mul'
)

scalar_ne = Operation(
    name='scalar_ne',
    defaults='myia.operations.prim_scalar_ne'
)

scalar_pow = Operation(
    name='scalar_pow',
    defaults='myia.operations.prim_scalar_pow'
)

scalar_sign = Operation(
    name='scalar_sign',
    defaults='myia.operations.prim_scalar_sign'
)

scalar_sin = Operation(
    name='scalar_sin',
    defaults='myia.operations.prim_scalar_sin'
)

scalar_sub = Operation(
    name='scalar_sub',
    defaults='myia.operations.prim_scalar_sub'
)

scalar_tan = Operation(
    name='scalar_tan',
    defaults='myia.operations.prim_scalar_tan'
)

scalar_tanh = Operation(
    name='scalar_tanh',
    defaults='myia.operations.prim_scalar_tanh'
)

scalar_to_array = Operation(
    name='scalar_to_array',
    defaults='myia.operations.prim_scalar_to_array'
)

scalar_trunc = Operation(
    name='scalar_trunc',
    defaults='myia.operations.prim_scalar_trunc'
)

scalar_uadd = Operation(
    name='scalar_uadd',
    defaults='myia.operations.prim_scalar_uadd'
)

scalar_usub = Operation(
    name='scalar_usub',
    defaults='myia.operations.prim_scalar_usub'
)

scatter = Operation(
    name='scatter',
    defaults='myia.operations.prim_scatter'
)

scatter_add = Operation(
    name='scatter_add',
    defaults='myia.operations.prim_scatter_add'
)

shape = Operation(
    name='shape',
    defaults='myia.operations.prim_shape'
)

slice = Operation(
    name='slice',
    defaults='myia.operations.op_slice'
)

split = Operation(
    name='split',
    defaults='myia.operations.prim_split'
)

stop_gradient = Operation(
    name='stop_gradient',
    defaults='myia.operations.prim_stop_gradient'
)

string_eq = Operation(
    name='string_eq',
    defaults='myia.operations.prim_string_eq'
)

string_ne = Operation(
    name='string_ne',
    defaults='myia.operations.ops_string.string_ne'
)

sub = Operation(
    name='sub',
    defaults='myia.operations.ops_dunder.sub'
)

sum = Operation(
    name='sum',
    defaults='myia.operations.ops_array.sum'
)

switch = Operation(
    name='switch',
    defaults='myia.operations.prim_switch'
)

t = Operation(
    name='t',
    defaults='myia.operations.ops_array.t'
)

tagged = Operation(
    name='tagged',
    defaults='myia.operations.prim_tagged'
)

take = Operation(
    name='take',
    defaults='myia.operations.prim_take'
)

take_grad_inp = Operation(
    name='take_grad_inp',
    defaults='myia.operations.prim_take_grad_inp'
)

to_scalar_type = Operation(
    name='to_scalar_type',
    defaults='myia.operations.macro_to_scalar_type'
)

transpose = Operation(
    name='transpose',
    defaults='myia.operations.prim_transpose'
)

truediv = Operation(
    name='truediv',
    defaults='myia.operations.ops_dunder.truediv'
)

trunc = Operation(
    name='trunc',
    defaults='myia.operations.ops_dunder.trunc'
)

tuple_concat = Operation(
    name='tuple_concat',
    defaults='myia.operations.ops_tuple.tuple_concat'
)

tuple_get = Operation(
    name='tuple_get',
    defaults='myia.operations.ops_tuple.tuple_get'
)

tuple_getitem = Operation(
    name='tuple_getitem',
    defaults='myia.operations.prim_tuple_getitem'
)

tuple_getslice = Operation(
    name='tuple_getslice',
    defaults='myia.operations.ops_tuple.tuple_getslice'
)

tuple_hasnext = Operation(
    name='tuple_hasnext',
    defaults='myia.operations.ops_tuple.tuple_hasnext'
)

tuple_len = Operation(
    name='tuple_len',
    defaults='myia.operations.macro_tuple_len'
)

tuple_next = Operation(
    name='tuple_next',
    defaults='myia.operations.ops_tuple.tuple_next'
)

tuple_setitem = Operation(
    name='tuple_setitem',
    defaults='myia.operations.prim_tuple_setitem'
)

typeof = Operation(
    name='typeof',
    defaults='myia.operations.macro_typeof'
)

universal = Operation(
    name='universal',
    defaults='myia.operations.macro_universal'
)

universe_getitem = Operation(
    name='universe_getitem',
    defaults='myia.operations.prim_universe_getitem'
)

universe_setitem = Operation(
    name='universe_setitem',
    defaults='myia.operations.prim_universe_setitem'
)

unsafe_static_cast = Operation(
    name='unsafe_static_cast',
    defaults='myia.operations.prim_unsafe_static_cast'
)

user_switch = Operation(
    name='user_switch',
    defaults='myia.operations.macro_user_switch'
)

value_and_grad = Operation(
    name='value_and_grad',
    defaults='myia.operations.op_value_and_grad'
)

xor = Operation(
    name='xor',
    defaults='myia.operations.ops_dunder.xor'
)

zeros_like = Operation(
    name='zeros_like',
    defaults='myia.operations.op_zeros_like'
)

zip = Operation(
    name='zip',
    defaults='myia.operations.op_zip'
)
