"""Abstract data and type/shape inference."""

from .loop import (  # noqa
    InferenceLoop,
    Pending,
    PendingFromList,
    PendingTentative,
    find_coherent_result,
    find_coherent_result_sync,
    force_pending,
)


from .utils import (  # noqa
    build_value,
    type_token,
    type_to_abstract,
    abstract_check,
    abstract_clone,
    abstract_clone_async,
    concretize_abstract,
    broaden,
    sensitivity_transform,
    amerge,
    bind,
    typecheck,
    split_type,
    hastype_helper,
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
    AbstractExternal,
    AbstractFunction,
    AbstractTuple,
    AbstractArray,
    AbstractList,
    AbstractClassBase,
    AbstractClass,
    AbstractADT,
    AbstractJTagged,
    AbstractUnion,
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
    Contextless,
    CONTEXTLESS,
    AbstractReference,
    Reference,
    VirtualReference,
    EvaluationCache,
)


from .infer import (  # noqa
    ArrayWrapper,
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
