
from .proj import natproj, getprop
from ..symbols import builtins
from ..interpret import Primitive
from ..inference.avm import AbstractValue, WrappedException
from ..inference.types import *
from .main import impl_bank


_ = True


##################
# Type inferrers #
##################


T = var('T')
N = var('N', Number)


numeric_bin = [
    (N, N, N),
    (Array[N], Array[N], Array[N])
]

numarray_bin = [
    (Array[N], Array[N], Array[N])
]

type_signatures = {
    builtins.add: numeric_bin,
    builtins.subtract: numeric_bin,
    # builtins.multiply: numeric_bin,
    # builtins.divide: numeric_bin,
    builtins.dot: numarray_bin,
    builtins.equal: (T, T, Bool),
    builtins.less: (N, N, Bool),
    builtins.greater: (N, N, Bool),
    builtins.switch: (Bool, T, T, T),
    builtins.identity: (T, T)
}


def std_type(sym, sigs):
    if not isinstance(sigs, list):
        sigs = [sigs]

    def check_type(*args):
        args = tuple(getprop(arg, builtins.type) for arg in args)
        for sig in sigs:
            *isig, osig = sig
            d = unify(isig, args)
            if d is not False:
                return reify(osig, d)
        raise WrappedException(TypeError(f'Type error ({sym}).'))

    return check_type


type_projs = impl_bank['project'].setdefault(builtins.type, {})


for sym, sigs in type_signatures.items():
    type_projs[impl_bank['abstract'][sym]] = \
        Primitive(std_type(sym, sigs))


@natproj(builtins.type)
def proj_mktuple(*args):
    def gettype(arg):
        if builtins.type in arg.values:
            return arg[builtins.type]
        else:
            return impl_bank['abstract'][builtins.type](arg)
    return Tuple[tuple(map(gettype, args))]
