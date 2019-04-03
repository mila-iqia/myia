
from myia.opt import lib


tuple_opts = [
    lib.getitem_tuple,
    lib.setitem_tuple,
    lib.bubble_op_tuple_binary,
]


arith_opts = [
    lib.multiply_by_zero_l,
    lib.multiply_by_zero_r,
    lib.multiply_by_one_l,
    lib.multiply_by_one_r,
    lib.add_zero_l,
    lib.add_zero_r,
]


all_opt = [
    *tuple_opts,
    *arith_opts,
    lib.inline,
]
