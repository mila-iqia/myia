"""Utilities to validate annotations after inference."""

import typing

import numpy as np
from ovld import ovld

from ..abstract import (
    AbstractADT,
    AbstractArray,
    AbstractDict,
    AbstractRandomState,
    AbstractScalar,
    AbstractTuple,
)
from ..utils import AnnotationMismatchError
from ..utils.misc import RandomStateWrapper
from ..xtype import pytype_to_myiatype


def _raise_error(annotation, abstract, message=""):
    """Raise an annotation error with formatted message."""
    if message:
        message = f": {message}"
    raise AnnotationMismatchError(
        f"Annotation {annotation} vs abstract {abstract}{message}"
    )


@ovld.dispatch
def _check(self, cls, args, abstract):
    return self[cls, object, object](cls, args, abstract)


@ovld  # noqa: F811
def _check(
    cls: (bool, type(None), str, int, float, np.integer, np.floating),
    _,
    abstract,
):
    if not isinstance(abstract, AbstractScalar):
        _raise_error(cls, abstract, "expected an AbstractScalar")
    expected_type = pytype_to_myiatype(cls)
    if abstract.xtype() != expected_type:
        _raise_error(cls, abstract, f"expected scalar type {expected_type}")


@ovld  # noqa: F811
def _check(cls: tuple, args, abstract):
    if not isinstance(abstract, AbstractTuple):
        _raise_error(cls, abstract, "expected an AbstractTuple")
    if args:
        if len(args) != len(abstract.elements):
            _raise_error(
                (cls, args),
                abstract,
                f"expected {len(args)} tuple types, got {len(abstract.elements)}",
            )
        for i, (element_type, element) in enumerate(
            zip(args, abstract.elements)
        ):
            validate_annotation(element_type, element)


@ovld  # noqa: F811
def _check(cls: list, args, abstract):
    if not isinstance(abstract, AbstractADT):
        _raise_error(cls, abstract, "expected an AbstractADT")
    if args:
        (element_type,) = args
        if abstract.attributes:
            validate_annotation(element_type, abstract.attributes["head"])


@ovld  # noqa: F811
def _check(cls: dict, args, abstract):
    if not isinstance(abstract, AbstractDict):
        _raise_error(cls, abstract, "expected an AbstractDict")
    if args:
        key_type, value_type = args
        if not isinstance(key_type, type):
            _raise_error(
                (cls, args), abstract, f"unsupported key type: {key_type}",
            )
        for key, value in abstract.entries.items():
            if not isinstance(key, key_type):
                _raise_error(
                    (cls, args),
                    abstract,
                    f"wrong key type, expected {key_type}, got {type(key)}",
                )
            validate_annotation(value_type, value)


@ovld  # noqa: F811
def _check(cls: np.ndarray, _, abstract):
    if not isinstance(abstract, AbstractArray):
        _raise_error(cls, abstract, "expected an AbstractArray")


@ovld  # noqa: F811
def _check(cls: RandomStateWrapper, _, abstract):
    if not isinstance(abstract, AbstractRandomState):
        _raise_error(cls, abstract, "expected an AbstractRandomState")


def validate_annotation(annotation, abstract):
    """Check abstract based on annotation."""
    if isinstance(annotation, typing._GenericAlias):
        cls = annotation.__origin__
        args = annotation.__args__
    else:
        cls = annotation
        args = ()
    _check(cls, args, abstract)
    return abstract
