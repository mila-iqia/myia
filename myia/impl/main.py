
from typing import Dict, Any
from collections import defaultdict
from ..symbols import builtins


impl_bank: Dict[str, Dict[Any, Any]] = defaultdict(dict)


def symbol_associator(kind):
    prefix = f'{kind}_'

    def associator_deco(process):
        def deco(fn):
            assert fn.__name__.startswith(prefix)
            fname = fn.__name__[len(prefix):]
            assert hasattr(builtins, fname)
            sym = getattr(builtins, fname)
            return process(sym, fname, fn)
        return deco
    return associator_deco
