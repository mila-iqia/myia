
from .main import symbol_associator, impl_bank
from ..interpret import PrimitiveImpl, FunctionImpl
from ..inference.avm import AbstractValue, VALUE, ERROR
from ..parse import parse_function


######################
# Projection helpers #
######################


def proj(psym):
    projectors = impl_bank['project']
    projs = projectors.setdefault(psym, {})

    @symbol_associator('proj')
    def pimpl(sym, name, fn):
        fsym, fenv = parse_function(fn)
        for s, lbda in fenv.bindings.items():
            impl_bank['abstract'][s] = \
                FunctionImpl(lbda, [impl_bank['abstract']])
        projs[impl_bank['abstract'][sym]] = \
            FunctionImpl(fenv[fsym], [impl_bank['abstract']])
        return fn

    return pimpl


def getprop(v, sym):
    fn = impl_bank['abstract'][sym]
    if isinstance(v, AbstractValue):
        if sym in v.values:
            return v[sym]
        elif ERROR in v.values:
            raise v[ERROR]
        else:
            return fn(v[VALUE])
    else:
        raise fn(v)


def natproj(psym):
    projectors = impl_bank['project']
    projs = projectors.setdefault(psym, {})

    @symbol_associator('proj')
    def pimpl(sym, name, fn):
        projs[impl_bank['abstract'][sym]] = \
            PrimitiveImpl(fn)
        return fn

    return pimpl
