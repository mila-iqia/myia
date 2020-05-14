"""Serialization utilities for graphs and their properties."""

# This is a mess of imports

# 1) We prefer ruamel.yaml since it has merged the large files fix.
# 2) Conda has a different name for it since it doesn't like namespaces.
# 3) We attempt to use pyyaml as a fallback
try:  # pragma: no cover
    try:
        from ruamel.yaml import (
            CSafeLoader as SafeLoader,
            CSafeDumper as SafeDumper,
        )
    except ImportError:
        try:
            from ruamel_yaml import (
                CSafeLoader as SafeLoader,
                CSafeDumper as SafeDumper,
            )
        except ImportError:
            from yaml import (
                CSafeLoader as SafeLoader,
                CSafeDumper as SafeDumper,
            )
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        """
Couldn't find a C-backed version of yaml.

Please install either ruamel.yaml or PyYAML with the C extension. The python
 versions are just too slow to work properly.
"""
    ) from e

import codecs
import os
import traceback


class MyiaDumper(SafeDumper):
    """Customize the dumper."""

    def __init__(self, fd):
        """Record stream, even for C."""
        stream = os.fdopen(fd, "wb", buffering=0, closefd=False)
        super().__init__(stream, encoding="utf-8", explicit_end=True)
        self.stream = stream

    def represent(self, data):
        """Represent and flush."""
        super().represent(data)
        self.stream.flush()


class MyiaLoader(SafeLoader):
    """Customize the loader."""

    def __init__(self, fd):
        """Make sure reads don't block."""
        stream = os.fdopen(fd, "rb", buffering=0, closefd=False)
        super().__init__(stream)

    def determine_encoding(self):  # pragma: no cover
        """This is a workaround for the python version of pyyaml.

        It really wants to read from the stream when creating the
        loader object to figure out the encoding.  We just statically
        figure it out here instead.
        """
        self.raw_decode = codecs.utf_8_decode
        self.encoding = "utf-8"
        self.raw_buffer = b""

    def construct_document(self, node):
        """Add support for finalizers."""
        self._finalizers = []
        res = super().construct_document(node)
        for f in self._finalizers:
            f()
        self._finalizers = []
        return res

    def add_finalizer(self, f):
        """Register a finalizer to be run when the loading is finished."""
        assert callable(f)
        self._finalizers.append(f)


class LoadedError(Exception):
    """Represent an error that was serialized."""

    def __init__(self, message):
        """An error with a single message."""
        self.message = message

    def __str__(self):
        return self.message

    def __repr__(self):
        return f"LoadedException"


def __serialize__(self, dumper):
    data = self._serialize()
    assert isinstance(data, dict)
    return dumper.represent_mapping(getattr(self, "@SERIAL_TAG"), data)


def _serialize(dumper, self):
    return self.__serialize__(dumper)


def _make_construct(cls):
    def _construct(loader, node):
        it = cls._construct()
        yield next(it)
        data = loader.construct_mapping(node)
        try:
            it.send(data)
        except StopIteration as e:
            if e.value is not None:
                loader.add_finalizer(e.value)

    return _construct


def _serialize_tuple(dumper, data):
    return dumper.represent_sequence("tuple", data)


def _construct_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


MyiaDumper.add_representer(tuple, _serialize_tuple)
MyiaLoader.add_constructor("tuple", _construct_tuple)


_OBJ_MAP = {}
_TAG_MAP = {}


def _serialize_unique(dumper, obj):
    tag = _OBJ_MAP.get(obj, None)
    if tag is None:
        if isinstance(obj, Exception):
            return dumper.represent_scalar(
                "error",
                "\n".join(
                    traceback.format_exception(
                        type(obj), obj, obj.__traceback__
                    )
                ),
            )
        else:
            return dumper.represent_undefined(obj)
    else:
        return dumper.represent_scalar(tag, "")


def _construct_unique(loader, node):
    if node.tag == "error":
        data = loader.construct_scalar(node)
        return LoadedError(data)
    obj = _TAG_MAP.get(node.tag, None)
    if obj is None:
        return loader.construct_undefined(node)  # pragma: no cover
    else:
        return obj


MyiaDumper.add_representer(None, _serialize_unique)
MyiaLoader.add_constructor(None, _construct_unique)


def register_serialize(obj, tag):
    """Serialize unique objects.

    The object instance will be associated with the given tag.
    """
    assert isinstance(tag, str)
    assert obj not in _OBJ_MAP
    assert tag not in _TAG_MAP
    _OBJ_MAP[obj] = tag
    _TAG_MAP[tag] = obj


def dump(o, fd):
    """Dump the passed-in object to the specified stream."""
    dumper = MyiaDumper(fd)
    try:
        dumper.open()
        dumper.represent(o)
        dumper.close()
    finally:
        dumper.dispose()


def load(fd):
    """Load one object from the stream."""
    loader = MyiaLoader(fd)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


__consolidate__ = True
__all__ = [
    "LoadedError",
    "MyiaDumper",
    "MyiaLoader",
    "dump",
    "load",
    "register_serialize",
]
