
from .ast import Symbol, Literal
from .util import Props


def _bsym(name):
    return Symbol(name, namespace='builtin')


builtins_dict = dict(
    add = _bsym('add'),
    subtract = _bsym('subtract'),
    multiply = _bsym('multiply'),
    divide = _bsym('divide'),
    power = _bsym('power'),
    dot = _bsym('dot'),
    bitwise_or = _bsym('bitwise_or'),
    bitwise_and = _bsym('bitwise_and'),
    bitwise_xor = _bsym('bitwise_xor'),
    unary_add = _bsym('unary_add'),
    unary_subtract = _bsym('unary_subtract'),
    bitwise_not = _bsym('bitwise_not'),
    negate = _bsym('negate'),
    less = _bsym('less'),
    greater = _bsym('greater'),
    less_equal = _bsym('less_equal'),
    greater_equal = _bsym('greater_equal'),
    equal = _bsym('equal'),
    range = _bsym('range'),
    index = _bsym('index'),
    map = _bsym('map'),
    filter = _bsym('filter'),
    getattr = _bsym('getattr'),
    setslice = _bsym('setslice')
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
