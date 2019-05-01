"""Abstract data and type/shape inference."""

from .loop import (  # noqa
    InferenceLoop,
    Pending,
    PendingFromList,
    PendingTentative,
    find_coherent_result,
    force_pending,
)


from .utils import (  # noqa
    build_value,
    build_type,
    build_type_fn,
    abstract_check,
    abstract_clone,
    abstract_clone_async,
    concretize_abstract,
    broaden,
    sensitivity_transform,
    amerge,
    bind,
)


from .data import (  # noqa
    ABSENT,
    ANYTHING,
    VOID,
    DEAD,
    POLY,
    Possibilities,
    PrimitiveFunction,
    GraphFunction,
    MetaGraphFunction,
    PartialApplication,
    JTransformedFunction,
    VirtualFunction,
    TypedPrimitive,
    DummyFunction,
    AbstractValue,
    AbstractScalar,
    AbstractType,
    AbstractError,
    AbstractFunction,
    AbstractTuple,
    AbstractArray,
    AbstractList,
    AbstractClass,
    AbstractJTagged,
    AbstractUnion,
    abstract_union,
    TrackDict,
    Track,
    VALUE,
    TYPE,
    SHAPE,
    infer_trace,
    Unspecializable,
    InferenceError,
    MyiaTypeError,
    MyiaShapeError,
    format_abstract,
    pretty_struct,
    pretty_type,
)


from .ref import (  # noqa
    Context,
    ConditionalContext,
    Contextless,
    CONTEXTLESS,
    AbstractReference,
    Reference,
    VirtualReference,
    EvaluationCache,
)


from .infer import (  # noqa
    Inferrer,
    BaseGraphInferrer,
    GraphInferrer,
    MetaGraphInferrer,
    TrackedInferrer,
    from_value,
    to_abstract,
    InferenceEngine,
)


from .prim import (  # noqa
    abstract_inferrer_constructors,
)
