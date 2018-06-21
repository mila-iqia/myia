"""Primitive operations.

Primitive operations are handled as constants in the intermediate
representation, with the constant's value being an instance of a `Primitive`
subclass.

"""


from ..utils import Named


class Primitive(Named):
    """Base class for primitives."""

    pass


##############
# Arithmetic #
##############


add = Primitive('add')
sub = Primitive('sub')
mul = Primitive('mul')
div = Primitive('div')
mod = Primitive('mod')
pow = Primitive('pow')
log = Primitive('log')
uadd = Primitive('uadd')
usub = Primitive('usub')


###############
# Comparisons #
###############


eq = Primitive('eq')
lt = Primitive('lt')
gt = Primitive('gt')
ne = Primitive('ne')
le = Primitive('le')
ge = Primitive('ge')
not_ = Primitive('not')


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


##########
# Arrays #
##########


shape = Primitive('shape')
map_array = Primitive('map_array')
scan_array = Primitive('scan_array')
reduce_array = Primitive('reduce_array')
distribute = Primitive('distribute')
reshape = Primitive('reshape')
dot = Primitive('dot')


##############
# Statements #
##############


if_ = Primitive('if')
return_ = Primitive('return')


#################
# Miscellaneous #
#################

maplist = Primitive('maplist')
resolve = Primitive('resolve')
partial = Primitive('partial')


##############################
# Gradient-related operations #
##############################

J = Primitive('J')
Jinv = Primitive('Jinv')
zeros_like = Primitive('zeros_like')
identity = Primitive('identity')
