"""
Defines a decorator for implementations. See myia.impl.impl_interp
for a use of ``symbol_associator``.

Take note of the ``impl_bank`` dictionary. It maps keys such as 'interp'
or 'abstract' to a set of implementations. This is where all implementations
are stored, in fact.
"""


from typing import Dict, Any
from collections import defaultdict
from ..symbols import builtins
from ..stx import globals_pool


impl_bank: Dict[str, Dict[Any, Any]] = defaultdict(dict)


def symbol_associator(kind):
    prefix = f'{kind}_' if kind else ''

    def associator_deco(process):
        def deco(fn):
            assert fn.__name__.startswith(prefix)
            fname = fn.__name__[len(prefix):]
            assert hasattr(builtins, fname)
            sym = getattr(builtins, fname)
            return process(sym, fname, fn)
        return deco
    return associator_deco
