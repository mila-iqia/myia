"""Primitive operations.

Primitive operations are handled as constants in the intermediate
representation, with the constant's value being an instance of a `Primitive`
subclass.

"""


from myia.utils import Named


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
exp = Primitive('exp')
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


##############
# Statements #
##############


if_ = Primitive('if')
return_ = Primitive('return')


##############################
# Gradient-related operations #
##############################

J = Primitive('J')
Jinv = Primitive('Jinv')
zeros_like = Primitive('zeros_like')
