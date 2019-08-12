"""General utilities."""

from .env import EnvInstance, SymbolicKeyInstance, newenv, smap  # noqa
from .errors import (  # noqa
    InferenceError,
    MyiaInputTypeError,
    MyiaShapeError,
    MyiaTypeError,
    TypeMismatchError,
    check_nargs,
    infer_trace,
    type_error_nargs,
)
from .intern import (  # noqa
    Atom,
    AttrEK,
    EqKey,
    IncompleteException,
    Interned,
    InternedMC,
    ItemEK,
    PossiblyRecursive,
    RecursionException,
    deep_eqkey,
    eq,
    eqkey,
    eqrec,
    hash,
    hashrec,
    intern,
)
from .merge import (  # noqa
    DELETE,
    Merge,
    MergeMode,
    Override,
    Reset,
    cleanup,
    merge,
)
from .misc import (  # noqa
    ADT,
    MISSING,
    NS,
    UNKNOWN,
    ClosureNamespace,
    Cons,
    Empty,
    ErrorPool,
    Event,
    Events,
    ModuleNamespace,
    Named,
    Namespace,
    Registry,
    Slice,
    TaggedValue,
    dataclass_methods,
    eprint,
    is_dataclass_type,
    keyword_decorator,
    list_str,
    list_to_cons,
    repr_,
)
from .orderedset import OrderedSet  # noqa
from .overload import Overload, TypeMap, overload, overload_wrapper  # noqa
from .partial import Partial, Partializable, partition_keywords  # noqa
from .profile import Profile, no_prof, print_profile  # noqa
from .unify import (  # noqa
    FilterVar,
    PredicateSet,
    RestrictedVar,
    Seq,
    SVar,
    Unification,
    UnionVar,
    Var,
    VisitError,
    expandlist,
    noseq,
    svar,
    uvar,
    var,
)
