"""General utilities."""

from .merge import (  # noqa
    DELETE, MergeMode, Merge, Reset, Override,
    merge, cleanup
)

from .misc import (  # noqa
    Named, UNKNOWN, Registry, repr_, list_str, TypeMap, StructuralMap, smap,
    Event, Events, NS, Namespace, ModuleNamespace, ClosureNamespace, eprint,
    is_dataclass_type, as_frozen, Overload, overload
)

from .partial import (  # noqa
    partition_keywords, Partial, Partializable
)

from .unify import (  # noqa
    Unification, Var, Seq, SVar, UnionVar, RestrictedVar, PredicateSet,
    FilterVar, var, svar, uvar, expandlist, noseq, VisitError
)
