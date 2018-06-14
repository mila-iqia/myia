"""General utilities."""

from .merge import (  # noqa
    DELETE, MergeMode, Merge, Reset, Override,
    merge, cleanup
)

from .misc import (  # noqa
    Named, Registry, repr_, list_str, TypeMap, StructuralMap, smap,
    Event, Events, NS, Namespace, ModuleNamespace, ClosureNamespace
)

from .partial import (  # noqa
    partition_keywords, Partial, Partializable
)
