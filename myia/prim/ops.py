"""Primitive operations.

Primitive operations are handled as constants in the intermediate
representation, with the constant's value being an instance of a `Primitive`
subclass.

"""


from ..utils import Named, serializable


@serializable('prim', scalar=True)
class Primitive(Named):
    """Base class for primitives."""

    def _serialize(self):
        return self.name

    @classmethod
    def _construct(cls, data):
        g = globals()
        p = g.get(data, None)
        if p is None:
            p = g[data + '_']
        assert isinstance(p, Primitive)
        return p


##############
# Arithmetic #
##############


scalar_add = Primitive('scalar_add')
scalar_sub = Primitive('scalar_sub')
scalar_mul = Primitive('scalar_mul')
scalar_div = Primitive('scalar_div')
scalar_mod = Primitive('scalar_mod')
scalar_pow = Primitive('scalar_pow')
scalar_trunc = Primitive('scalar_trunc')
scalar_floor = Primitive('scalar_floor')
scalar_max = Primitive('scalar_max')
scalar_uadd = Primitive('scalar_uadd')
scalar_usub = Primitive('scalar_usub')
scalar_exp = Primitive('scalar_exp')
scalar_log = Primitive('scalar_log')
scalar_sin = Primitive('scalar_sin')
scalar_cos = Primitive('scalar_cos')
scalar_tan = Primitive('scalar_tan')
scalar_tanh = Primitive('scalar_tanh')


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
bool_eq = Primitive('bool_eq')
string_eq = Primitive('string_eq')


##########
# Typing #
##########


hastype = Primitive('hastype')
tagged = Primitive('tagged')
unsafe_static_cast = Primitive('unsafe_static_cast')
scalar_cast = Primitive('scalar_cast')
hastag = Primitive('hastag')
casttag = Primitive('casttag')


###################
# Data structures #
###################


make_tuple = Primitive('make_tuple')
tuple_getitem = Primitive('tuple_getitem')
tuple_setitem = Primitive('tuple_setitem')

make_dict = Primitive('make_dict')
dict_getitem = Primitive('dict_getitem')
dict_setitem = Primitive('dict_setitem')

make_record = Primitive('make_record')
record_getitem = Primitive('record_getitem')
record_setitem = Primitive('record_setitem')

array_getitem = Primitive('array_getitem')
array_setitem = Primitive('array_setitem')


##########
# Arrays #
##########


scalar_to_array = Primitive('scalar_to_array')
array_to_scalar = Primitive('array_to_scalar')
broadcast_shape = Primitive('broadcast_shape')
invert_permutation = Primitive('invert_permutation')
shape = Primitive('shape')
array_map = Primitive('array_map')
array_scan = Primitive('array_scan')
array_reduce = Primitive('array_reduce')
distribute = Primitive('distribute')
reshape = Primitive('reshape')
transpose = Primitive('transpose')
dot = Primitive('dot')

conv2d = Primitive('conv2d')
conv2d_input_grad = Primitive('conv2d_input_grad')
conv2d_weight_grad = Primitive('conv2d_weight_grad')


##############
# Statements #
##############


switch = Primitive('switch')
return_ = Primitive('return')
raise_ = Primitive('raise')


#################
# Miscellaneous #
#################


identity = Primitive('identity')
partial = Primitive('partial')
J = Primitive('J')
Jinv = Primitive('Jinv')
env_setitem = Primitive('env_setitem')
env_getitem = Primitive('env_getitem')
env_add = Primitive('env_add')
exception = Primitive('exception')
make_kwarg = Primitive('make_kwarg')
extract_kwarg = Primitive('extract_kwarg')
