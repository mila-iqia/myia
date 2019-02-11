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
    AbstractBase,
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
