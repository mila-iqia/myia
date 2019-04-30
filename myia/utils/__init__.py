"""General utilities."""

from .intern import (  # noqa
    Interned, InternedMC, EqKey, Atom, Elements, eqkey, deep_eqkey,
    RecursionException, eq, hash, hashrec, eqrec, IncompleteException
)

from .merge import (  # noqa
    DELETE, MergeMode, Merge, Reset, Override,
    merge, cleanup
)

from .misc import (  # noqa
    Named, UNKNOWN, Registry, repr_, list_str, smap,
    Event, Events, NS, Namespace, ModuleNamespace, ClosureNamespace, eprint,
    is_dataclass_type, as_frozen, ErrorPool, flatten,
    SymbolicKeyInstance, EnvInstance, newenv
)

from .overload import (  # noqa
    TypeMap, Overload, overload, overload_wrapper
)

from .partial import (  # noqa
    partition_keywords, Partial, Partializable
)

from .profile import Profile, no_prof, print_profile  # noqa

from .unify import (  # noqa
    Unification, Var, Seq, SVar, UnionVar, RestrictedVar, PredicateSet,
    FilterVar, var, svar, uvar, expandlist, noseq, VisitError
)

from .orderedset import OrderedSet  # noqa
