"""Utilities to validate annotations after inference."""

import inspect
import typing

import numpy as np

from myia.validate import ValidationError

from ..abstract import (
    AbstractADT,
    AbstractArray,
    AbstractDict,
    AbstractRandomState,
    AbstractScalar,
    AbstractTuple,
    AbstractUnion,
)
from ..classes import Empty
from ..ir.anf import ANFNode
from ..utils.misc import RandomStateWrapper
from ..xtype import pytype_to_myiatype


class AnnotationValidationError(ValidationError):
    """Exception raised when an annotation error occurs."""


def _raise_error(node_name, annotation, abstract, message=""):
    """Raise an annotation error with formatted message."""
    if message:
        message = f": {message}"
    raise AnnotationValidationError(
        f"Node {node_name}, annotation {annotation} vs abstract {abstract}{message}"
    )


class _Introspection:
    """Helper class to collect validation functions for many annotations.

    Like ovld, but ovld takes only objects while here we need types.
    """

    __slots__ = ("mapping",)

    def __init__(self):
        """Initialize."""
        self.mapping = {}

    @staticmethod
    def __is_typing(typedef):
        """Return True if typedef is a type hinting from typing module."""
        return isinstance(typedef, typing._GenericAlias)

    @staticmethod
    def __is_type(typedef):
        """Return True if typedef is a type (e.g. int)."""
        return isinstance(typedef, type)

    def register(self, *type_definitions):
        """Decorator to register a validator for given type definitions.

        A type definition must be either a type or a type hint from
        typing module.

        A validator must be a function which will receive 2 special
        arguments as first:
        function(cls, sub_cls, *args, **kwargs)
        where:
        - cls is the type this validator must checks (e.g., list, tuple, int).
        - sub_cls is a tuple of type definitions which specializes cls.

        For example, if validator is called for type int, then
        cls is int and sub_cls is ()

        If validator is called for type int List[int], then
        cls is list and sub_cls is (int,)
        """

        def decorator(function):
            for typedef in type_definitions:
                if self.__is_type(typedef) or self.__is_typing(typedef):
                    self.mapping[typedef] = function
                else:
                    raise ValueError(f"Invalid typing: {typedef}")

        return decorator

    def __call__(self, typedef, *args, **kwargs):
        """Call a registered validator for given typedef.

        cls and sub_cls will be inferred from typedef and passed
        to validator with other parameters args and kwargs:

        validator(cls, sub_cls, *args, **kwargs)
        """

        function = None
        if self.__is_type(typedef):
            cls = typedef
            sub_cls = ()
            for parent_class in inspect.getmro(typedef):
                if parent_class in self.mapping:
                    function = self.mapping[parent_class]
                    break
        elif self.__is_typing(typedef):
            cls = typedef.__origin__
            sub_cls = typedef.__args__
            if typedef in self.mapping:
                function = self.mapping[typedef]
            elif cls in self.mapping:
                function = self.mapping[cls]
        else:
            raise ValueError(f"Invalid typing: {typedef}")

        if not function:
            raise ValueError(f"Unregistered typing: {typedef}")

        return function(cls, sub_cls, *args, **kwargs)


annotation_checker = _Introspection()


@annotation_checker.register(
    bool, type(None), str, int, float, np.integer, np.floating
)
def _check_scalar(cls, _, abstract, node_name):
    if not isinstance(abstract, AbstractScalar):
        _raise_error(node_name, cls, abstract, "expected an AbstractScalar")
    expected_type = pytype_to_myiatype(cls)
    if abstract.xtype() != expected_type:
        _raise_error(
            node_name, cls, abstract, f"expected scalar type {expected_type}"
        )


@annotation_checker.register(tuple)
def _check_tuple(cls, sub_cls, abstract, node_name):
    if not isinstance(abstract, AbstractTuple):
        _raise_error(node_name, cls, abstract, "expected an AbstractTuple")
    if sub_cls:
        if len(sub_cls) != len(abstract.elements):
            _raise_error(
                node_name,
                (cls, sub_cls),
                abstract,
                f"expected {len(sub_cls)} tuple types, got {len(abstract.elements)}",
            )
        for i, (element_type, element) in enumerate(
            zip(sub_cls, abstract.elements)
        ):
            annotation_checker(element_type, element, f"{node_name}[{i}]")


@annotation_checker.register(list)
def _check_list(cls, sub_cls, abstract, node_name):
    if not isinstance(abstract, AbstractADT):
        _raise_error(node_name, cls, abstract, "expected an AbstractADT")
    if sub_cls:
        if len(sub_cls) != 1:
            _raise_error(
                node_name,
                (cls, sub_cls),
                abstract,
                "expected only 1 element type for a list",
            )
        (element_type,) = sub_cls
        if abstract.attributes:
            if "head" not in abstract.attributes:
                _raise_error(
                    node_name,
                    cls,
                    abstract,
                    "missing head attribute in AbstractADT",
                )
            if "tail" not in abstract.attributes:
                _raise_error(
                    node_name,
                    cls,
                    abstract,
                    "missing tail attribute in AbstractADT",
                )
            annotation_checker(
                element_type, abstract.attributes["head"], f"{node_name}[0]"
            )
            if not isinstance(abstract.attributes["tail"], AbstractUnion):
                _raise_error(
                    node_name,
                    cls,
                    abstract,
                    "expected tail to be an AbstractUnion",
                )
            tail_options = abstract.attributes["tail"].options
            if len(tail_options) != 2:
                _raise_error(
                    node_name,
                    cls,
                    abstract,
                    f"expected tail to have 2 options, got {len(tail_options)}",
                )
            if (
                not isinstance(tail_options[0], AbstractADT)
                or tail_options[0].tag is not Empty
            ):
                _raise_error(
                    node_name,
                    cls,
                    abstract,
                    f"expected tail first option to be AbstractADT[Empty], got {tail_options[0]}",
                )
            if tail_options[1] != abstract:
                _raise_error(
                    node_name,
                    cls,
                    abstract,
                    f"expected tail 2nd option to be same as abstract, got {tail_options[1]}",
                )


@annotation_checker.register(dict)
def _check_dict(cls, sub_cls, abstract, node_name):
    if not isinstance(abstract, AbstractDict):
        _raise_error(node_name, cls, abstract, "expected an AbstractDict")
    if sub_cls:
        if len(sub_cls) != 2:
            _raise_error(
                node_name,
                (cls, sub_cls),
                abstract,
                f"expected 2 sub-types for a dict, got {len(sub_cls)}",
            )
        key_type, value_type = sub_cls
        if not isinstance(key_type, type):
            _raise_error(
                node_name,
                (cls, sub_cls),
                abstract,
                f"unsupported key type: {key_type}",
            )
        for key, value in abstract.entries.items():
            if not isinstance(key, key_type):
                _raise_error(
                    node_name,
                    (cls, sub_cls),
                    abstract,
                    f"wrong key type, expected {key_type}, got {type(key)}",
                )
            annotation_checker(value_type, value, f"{node_name}[{key}]")


@annotation_checker.register(np.ndarray)
def _check_ndarray(cls, _, abstract, node_name):
    if not isinstance(abstract, AbstractArray):
        _raise_error(node_name, cls, abstract, "expected an AbstractArray")


@annotation_checker.register(RandomStateWrapper)
def _check_rstate(cls, _, abstract, node_name):
    if not isinstance(abstract, AbstractRandomState):
        _raise_error(
            node_name, cls, abstract, "expected an AbstractRandomState"
        )


def validate_node(node: ANFNode):
    """Check abstract based on annotation (if available) for given node."""
    node_name = node.debug.debug_name
    annotation = node.annotation
    abstract = node.abstract
    if annotation:
        annotation_checker(annotation, abstract, node_name)
