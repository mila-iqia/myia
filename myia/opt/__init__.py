"""Optimization submodule."""

from .opt import (  # noqa
    VarNode, sexp_to_node, sexp_to_graph,
    PatternSubstitutionOptimization,
    NodeMap, pattern_replacer,
    LocalPassOptimizer,
    GraphTransform,
)

from .cse import (  # noqa
    cse,
    CSE,
)

from .clean import (  # noqa
    type_to_tag, simplify_types
)


from .dde import (  # noqa
    DeadDataElimination,
)
