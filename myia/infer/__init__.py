"""Inference engine (types, values, etc.)."""

from .core import (  # noqa
    InferenceLoop,
    EvaluationCache,
    EquivalenceChecker,
    InferenceVar,
    find_coherent_result,
    reify,
)

from .graph_infer import (  # noqa
    Context,
    Contextless,
    CONTEXTLESS,
    Track,
    InferenceEngine,
    AbstractReference,
    Reference,
    VirtualReference,
    TransformedReference,
    Inferrer,
    DummyInferrer,
    GraphInferrer,
    MetaGraphInferrer,
    MultiInferrer,
    PrimitiveInferrer,
    PartialInferrer,
    ExplicitInferrer,
    register_inferrer,
    concretize_type,
)

from .utils import (  # noqa
    ANYTHING,
    InferenceError,
    MyiaTypeError,
    MyiaShapeError,
    ValueWrapper,
    unwrap,
    Unspecializable,
    VOID,
    DEAD,
    POLY,
    AMBIGUOUS,
    INACCESSIBLE,
)
