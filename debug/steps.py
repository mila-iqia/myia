
from myia.utils import Partial
from myia.pipeline.steps import (
    step_parse as parse,
    step_resolve as resolve,
    step_infer as infer,
    step_specialize as specialize,
    step_erase_class as erase_class,
    step_opt as opt,
    step_debug_opt as debug_opt,
    step_erase_tuple as erase_tuple,
    step_opt2 as opt2,
    step_validate as validate,
    step_cconv as cconv,
    step_compile as export,
)

standard = [
    parse, resolve, infer, specialize,
    erase_class, opt, erase_tuple, opt2,
    validate, cconv, export
]

_debug_opt = [
    parse, resolve, infer, specialize,
    erase_class, debug_opt, erase_tuple, opt2,
    validate, cconv, export
]

_bang_parse = standard[:standard.index(parse) + 1]
_bang_resolve = standard[:standard.index(resolve) + 1]
_bang_infer = standard[:standard.index(infer) + 1]
_bang_specialize = standard[:standard.index(specialize) + 1]
_bang_erase_class = standard[:standard.index(erase_class) + 1]
_bang_opt = standard[:standard.index(opt) + 1]
_bang_erase_tuple = standard[:standard.index(erase_tuple) + 1]
_bang_opt2 = standard[:standard.index(opt2) + 1]
_bang_validate = standard[:standard.index(validate) + 1]
_bang_cconv = standard[:standard.index(cconv) + 1]
_bang_export = standard[:standard.index(export) + 1]

_bang_debug_opt = _debug_opt[:_debug_opt.index(debug_opt) + 1]


def _adjust():
    for name, g in globals().items():
        if isinstance(g, Partial):
            g._name = name


_adjust()
