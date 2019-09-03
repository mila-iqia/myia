"""Optimization submodule."""

from .clean import _strmap_tag, simplify_types, str_to_tag, type_to_tag  # noqa
from .cse import CSE, cse  # noqa
from .dde import DeadDataElimination  # noqa
from .opt import (  # noqa
    GraphTransform,
    LocalPassOptimizer,
    NodeMap,
    PatternSubstitutionOptimization,
    pattern_replacer,
)
