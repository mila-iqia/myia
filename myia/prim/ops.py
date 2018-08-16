"""Primitive operations.

Primitive operations are handled as constants in the intermediate
representation, with the constant's value being an instance of a `Primitive`
subclass.

"""


from ..utils import Named


class Primitive(Named):
    """Base class for primitives."""


##############
# Arithmetic #
##############


scalar_add = Primitive('scalar_add')
scalar_sub = Primitive('scalar_sub')
scalar_mul = Primitive('scalar_mul')
scalar_div = Primitive('scalar_div')
scalar_mod = Primitive('scalar_mod')
scalar_pow = Primitive('scalar_pow')
scalar_uadd = Primitive('scalar_uadd')
scalar_usub = Primitive('scalar_usub')


###############
# Comparisons #
###############


scalar_eq = Primitive('scalar_eq')
scalar_lt = Primitive('scalar_lt')
scalar_gt = Primitive('scalar_gt')
scalar_ne = Primitive('scalar_ne')
scalar_le = Primitive('scalar_le')
scalar_ge = Primitive('scalar_ge')
bool_not = Primitive('bool_not')
bool_and = Primitive('bool_and')
bool_or = Primitive('bool_or')


######################
# Type introspection #
######################


typeof = Primitive('typeof')
hastype = Primitive('hastype')


###################
# Data structures #
###################


cons_tuple = Primitive('cons_tuple')
head = Primitive('head')
tail = Primitive('tail')
getitem = Primitive('getitem')
setitem = Primitive('setitem')
getattr = Primitive('getattr')
setattr = Primitive('setattr')


#############
# Iteration #
#############


iter = Primitive('iter')
hasnext = Primitive('hasnext')
next = Primitive('next')


##########
# Arrays #
##########


scalar_to_array = Primitive('scalar_to_array')
broadcast_shape = Primitive('broadcast_shape')
shape = Primitive('shape')
array_map = Primitive('array_map')
array_map2 = Primitive('array_map2')
array_scan = Primitive('array_scan')
array_reduce = Primitive('array_reduce')
distribute = Primitive('distribute')
reshape = Primitive('reshape')
dot = Primitive('dot')


##############
# Statements #
##############


if_ = Primitive('if')
switch = Primitive('switch')
return_ = Primitive('return')


#################
# Miscellaneous #
#################

list_map = Primitive('list_map')
identity = Primitive('identity')
resolve = Primitive('resolve')
partial = Primitive('partial')
make_record = Primitive('make_record')
