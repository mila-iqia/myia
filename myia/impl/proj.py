
from .main import symbol_associator, impl_bank
from ..interpret import PrimitiveImpl, FunctionImpl
from ..inference.avm import AbstractValue, VALUE, ERROR
from ..parse import parse_function
from .impl_abstract import abstract_globals


######################
# Projection helpers #
######################


def proj(psym):
    projectors = impl_bank['project']
    projs = projectors.setdefault(psym, {})

    @symbol_associator('proj')
    def pimpl(sym, name, fn):
        lbda = parse_function(fn)
        projs[impl_bank['abstract'][sym]] = \
            FunctionImpl(lbda, abstract_globals)
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
