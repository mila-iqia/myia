
import inspect
from types import FunctionType
from typing import Tuple, Type, Any, Dict, Union


class Symbol:
    """
    Represent a variable name in Myia's frontend AST.

    Symbols should not be created directly. They should be created
    through a GenSym factory: GenSym enforces a unique namespace and
    keeps track of versions to guarantee that no Symbols accidentally
    collide.

    Attributes:
        label (str or Symbol): the name of the variable. If
            relation is None, this must be a string, otherwise
            this must be a Symbol.
        namespace (str): the namespace in which the variable
            lives. This is usually 'global', 'builtin', or a
            uuid created on a per-LambdaNode expression basis.
        version (int): differentiates variables with the same
            name and namespace. This can happen when there are
            multiple writes to the same variable in Python.
        relation (str): how this variable relates to some other
            variable in the 'label' attribute. For example,
            automatic differentiation will accumulate the gradient
            for variable x in a Symbol with label x and relation
            'sensitivity'.

    The HTML pretty-printer will show the version as a subscript
    (except for version 1), and the relation as a prefix on
    the representation of the parent Symbol.
    """
    def __init__(self,
                 label: Union[str, 'Symbol'],
                 *,
                 namespace: str = None,
                 version: int = 1,
                 relation: str = None) -> None:
        if relation is None:
            assert isinstance(label, str)
        else:
            assert isinstance(label, Symbol)
        self.label = label
        self.namespace = namespace
        self.version = version
        self.relation = relation

    def __str__(self) -> str:
        v = f'#{self.version}' if self.version > 1 else ''
        r = f'{self.relation}' if self.relation else ''
        return f'{r}{self.label}{v}'

    def __eq__(self, obj) -> bool:
        """Two symbols are equal if they have the same label,
        namespace, version and relation to their label."""
        s: Symbol = obj
        return isinstance(s, Symbol) \
            and self.label == s.label \
            and self.namespace == s.namespace \
            and self.version == s.version \
            and self.relation == s.relation

    def __hash__(self) -> int:
        return hash((self.label, self.namespace,
                     self.version, self.relation))


class Universe:
    """
    A lazy mapping from any Python value to values that conform to this
    `Universe`'s rules.

    The `acquire` function should be overriden to process its input in the
    desired way.

    Attributes:
        parent: A (optional) `Universe` that will transform items before they
            are given to `acquire`.
        cache: A mapping from cachable values to their resolved/transformed
            versions (through `acquire`).
    """
    @staticmethod
    def cachable() -> Tuple[Type, ...]:
        """
        Return a tuple of types for which `Universe` will cache the result of
        the transformation.
        """
        return ()

    def __init__(self, parent: 'Universe' = None) -> None:
        self.parent = parent
        self.cache: Dict[Any, Any] = {}

    def acquire(self, x: Any) -> Any:
        raise NotImplementedError()

    def __getitem__(self, item: Any) -> Any:
        """
        Map a Python object to a value from this `Universe`. If the value has
        a `cachable` type, the result will be cached for future uses.

        Arguments:
            item: The value to map through this `Universe`.
        """
        if self.parent is not None:
            item = self.parent[item]
        if isinstance(item, self.cachable()):
            try:
                return self.cache[item]
            except KeyError:
                v = self.acquire(item)
                self.cache[item] = v
                return v
        else:
            return self.acquire(item)


class PythonUniverse(Universe):
    """
    Resolves global variables and Myia-transformed objects to their originals.

    * A `Symbol` that points to a global variable is resolved to the value of
      that global variable
    * An object that has the ``__myia_base__`` attribute is mapped to the value
      of that attribute.
    * Any other value is mapped to itself.

    Globals dictionaries may be added manually via `add_source`.

    `PythonUniverse` does not cache any mapping.
    """
    def __init__(self):
        super().__init__()
        self.sources = {}

    def add_source(self, namespace: str, contents: Dict[str, Any]) -> None:
        if namespace not in self.sources:
            self.sources[namespace] = contents

    def acquire(self, x: Any) -> Any:
        if isinstance(x, Symbol):
            sym = x
            lbl = sym.label
            assert isinstance(lbl, str)
            try:
                globs = self.sources[sym.namespace]
            except KeyError as err:
                raise NameError(f'PythonUniverse cannot resolve {lbl}'
                                f' because it does not have access'
                                f' to namespace {sym.namespace}. Add it'
                                f' using the add_source method.') from None
            try:
                result = globs[lbl]
            except KeyError as err:
                builtins = globs['__builtins__']
                try:
                    if isinstance(builtins, dict):
                        # I don't know why this ever happens, but it does.
                        result = builtins[lbl]
                    else:
                        result = getattr(builtins, lbl)
                except (KeyError, AttributeError):
                    raise NameError(f"Could not resolve global: '{sym}' "
                                    f"in namespace: '{sym.namespace}'.") from None
            # We recursively acquire the result in case it is something we
            # need to process further.
            return self.acquire(result)
        elif hasattr(x, '__myia_base__'):
            return x.__myia_base__
        elif isinstance(x, FunctionType):
            fn = x
            filename = inspect.getfile(fn)
            self.add_source(f'global:{filename}', fn.__globals__)
            return x
        # TODO: tuples, records etc. must be looped over to be acquired
        else:
            return x


class PipelineGenerator:
    """
    Create pipelines of `Universe`s.

    `PipelineGenerator` caches `Universe`s so that different pipelines can share
    as much as possible: if two pipelines are created that have the same first
    three steps, for example (same type, same configuration), they will use the
    same `Universe`s and therefore the same cache for these steps.

    Args:
        constructors: A mapping from names to subclasses of `Universe`, or
            functions that generate `Universe`s.
    """
    def __init__(self, **constructors):
        self.constructors = {name: gen for name, gen in constructors.items()
                             if not isinstance(gen, str)}
        self.names = set(self.constructors.keys())
        self.cache = {}

    def get_pipeline(self, pipeline, cache):
        pipeline = tuple(pipeline)
        key = repr(pipeline)
        if cache and key in self.cache:
            return self.cache[key]

        *prev, (p, cfg) = pipeline
        if len(prev) > 0:
            cfg['parent'] = self.get_pipeline(prev, cache)
        u = self.constructors[p](**cfg)
        if cache:
            self.cache[key] = u
        return u

    def __call__(self, pipeline, cache=True, **config):
        """
        Generate a pipeline of universes.

        Args:
            pipeline: A -> delimited list of universes. The first universe
                in the list will have no parent, and each successive universe
                will have the previous one as a parent.
            cache: Whether to use a previously cached pipeline if one has already
                been made with the same configuration, or cache this pipeline.
                If `cache` is false, the returned value will be completely
                independent from any other pipeline.
            config: Configuration dictionary for each step in the pipeline.
                If config is not available for a step, an empty dictionary
                will be provided to its constructor.
        """
        path = pipeline.split('->')
        for name in path:
            if name not in self.names:
                raise NameError(f"'{name}' is not a valid Universe name."
                                f' Please pick from the set: {self.names}.')
        for name in config.keys():
            if name not in path:
                raise NameError(f"You provided config for '{name}', which is"
                                f' not in the pipeline \'{"->".join(path)}\'')
        pipeline = [(p, config.get(p, {})) for p in path]
        return self.get_pipeline(pipeline, cache)
