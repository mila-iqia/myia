try:
    from yaml import CSafeLoader as SafeLoader, CSafeDumper as SafeDumper
except ImportError:
    from yaml import SafeLoader, SafeDumper

import sys
from dataclasses import is_dataclass

# Should probably make a subclass of SafeDumper that is less retarded at
# dealing with represent_data for types (from dtype) and Named(s).
# Maybe I can hook add_representer(None, ...) for a fallback function.


def __serialize__(self, dumper):
    data = self._serialize()
    assert isinstance(data, dict)
    return dumper.represent_mapping(getattr(self, '@SERIAL_TAG'), data)


def __serialize_seq__(self, dumper):
    seq = self._serialize()
    assert isinstance(seq, (tuple, list))
    return dumper.represent_sequence(getattr(self, '@SERIAL_TAG'), seq)


def __serialize_scal__(self, dumper):
    scal = self._serialize()
    assert scal is not None
    return dumper.represent_scalar(getattr(self, '@SERIAL_TAG'), scal)


def __serialize_dc__(self, dumper):
    data = dict((k, getattr(self, k))
                for k in self.__dataclass_fields__.keys())
    return dumper.represent_mapping(getattr(self, '@SERIAL_TAG'), data)


def _serialize(dumper, self):
    return self.__serialize__(dumper)


def _make_construct(cls, dc, sequence, scalar):
    def _construct(loader, node):
        it = cls._construct()
        yield next(it)
        data = loader.construct_mapping(node)
        try:
            it.send(data)
        except StopIteration as e:
            return e.value
    if dc:
        if cls.__dataclass_params__.frozen:
            def _construct(loader, node):  # noqa: F811
                data = loader.construct_mapping(node)
                return cls(**data)
        else:
            breakpoint()
    if sequence:
        def _construct(loader, node):  # noqa: F811
            it = cls._construct()
            yield next(it)
            data = loader.construct_sequence(node)
            try:
                it.send(data)
            except StopIteration as e:
                return e.value
    if scalar:
        def _construct(loader, node):  # noqa: F811
            data = loader.construct_scalar(node)
            return cls._construct(data)
    return _construct


def serializable(tag, *, sequence=False, scalar=False):
    """Class decorator to make the wrapped class serializable.

    Parameters:
        tag: string, serialization tag, must be unique
        sequence: _serialize returns a sequence (tuple or list)
        scalar: _serialize returns a single item.

    """
    def wrapper(cls):
        dc = False
        if is_dataclass(cls):
            dc = True
        setattr(cls, '@SERIAL_TAG', tag)
        if not hasattr(cls, '__serialize__'):
            method = __serialize__
            if dc:
                method = __serialize_dc__
            if sequence:
                method = __serialize_seq__
            if scalar:
                method = __serialize_scal__
            setattr(cls, '__serialize__', method)
        _construct = _make_construct(cls, dc, sequence, scalar)
        SafeDumper.add_representer(cls, _serialize)
        SafeLoader.add_constructor(tag, _construct)
        return cls
    return wrapper


_OBJ_MAP = {}
_TAG_MAP = {}


def _serialize_unique(dumper, obj):
    tag = _OBJ_MAP.get(obj, None)
    if tag is None:
        return dumper.represent_undefined(obj)
    else:
        return dumper.represent_scalar(tag, '')


def _construct_unique(loader, node):
    obj = _TAG_MAP.get(node.tag, None)
    if obj is None:
        return loader.construct_undefined(node)
    else:
        return obj


SafeDumper.add_representer(None, _serialize_unique)
SafeLoader.add_constructor(None, _construct_unique)


def register_serialize(obj, tag):
    """Serialize unique objects.

    The object instance will be associated with the given tag.
    """
    assert isinstance(tag, str)
    assert obj not in _OBJ_MAP
    assert tag not in _TAG_MAP
    _OBJ_MAP[obj] = tag
    _TAG_MAP[tag] = obj


def dump(o, stream=sys.stdout):
    dumper = SafeDumper(stream)
    try:
        dumper.open()
        dumper.represent(o)
        dumper.close()
    finally:
        dumper.dispose()


def load(stream):
    loader = SafeLoader(stream)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()
