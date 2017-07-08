
from .ast import Symbol, Literal


class Props:
    def __init__(self, d):
        self.__dict__ = d

builtins_dict = dict(
    add = Symbol('add'),
    subtract = Symbol('subtract'),
    multiply = Symbol('multiply'),
    divide = Symbol('divide'),
    power = Symbol('power'),
    dot = Symbol('dot'),
    bitwise_or = Symbol('bitwise_or'),
    bitwise_and = Symbol('bitwise_and'),
    bitwise_xor = Symbol('bitwise_xor'),
    unary_add = Symbol('unary_add'),
    unary_subtract = Symbol('unary_subtract'),
    bitwise_not = Symbol('bitwise_not'),
    negate = Symbol('negate'),
    less = Symbol('less'),
    greater = Symbol('greater'),
    less_equal = Symbol('less_equal'),
    greater_equal = Symbol('greater_equal'),
    equal = Symbol('equal'),
    range = Symbol('range'),
    index = Symbol('index'),
    map = Symbol('map'),
    filter = Symbol('filter')
)

builtins = Props(builtins_dict)

operator_map = dict(
    Add = builtins.add,
    Sub = builtins.subtract,
    Mult = builtins.multiply,
    Div = builtins.divide,
    Pow = builtins.power,
    MatMult = builtins.dot,
    BitOr = builtins.bitwise_or,
    BitAnd = builtins.bitwise_and,
    BitXor = builtins.bitwise_xor,
    UAdd = builtins.unary_add,
    USub = builtins.unary_subtract,
    Invert = builtins.bitwise_not,
    Not = builtins.negate,
    Lt = builtins.less,
    Gt = builtins.greater,
    LtE = builtins.less_equal,
    GtE = builtins.greater_equal,
    Eq = builtins.equal
    # NotEq = builtins.
    # In = builtins.
    # NotIn = builtins.
    # Is = builtins.
    # IsNot = builtins.
)


function_map = {
    range: builtins.range,
}


# Not yet used [[[BEGIN

_maps = {
    'builtin': True,
    'numpy': False
}


def _add_numpy_map():
    # Note: we will only run this if numpy has already been
    # imported by the user
    import numpy as _
    numpy_map = {
        _.add: builtins.add,
        _.arange: builtins.range,
        _.divide: builtins.divide,
        _.dot: builtins.dot,
        _.multiply: builtins.multiply,
        _.subtract: builtins.subtract
    }
    numpy_map[_] = Literal(_)

    global function_map
    function_map = {**function_map, **numpy_map}


def _update_function_map():
    # This populates function_map with <function> => <Symbol> mappings,
    # e.g. numpy.add => Symbol("+")
    # However, we don't want to add mappings for a package if that package
    # has not been imported by the user, so we first check if it is present in
    # sys.modules, then we call the corresponding _add function defined above.
    for package, added in _maps.items():
        if not added and sys.modules[package]:
            # TODO: error handling
            globals()['_add_{}_map'.format(package)]()
            _maps[package] = True

# END]]]

def get_operator(node):
    try:
        return operator_map[node.__class__.__name__]
    except KeyError:
        raise NotImplementedError("Unknown operator: {}".format(node))


