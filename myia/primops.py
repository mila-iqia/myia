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


make_tuple = Primitive('make_tuple')
getitem = Primitive('getitem')
setitem = Primitive('setitem')


##############
# Statements #
##############


if_ = Primitive('if')
return_ = Primitive('return')
