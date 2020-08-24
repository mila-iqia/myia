"""Miscellaneous utilities."""

from dataclasses import is_dataclass

from ovld import ovld

from .misc import dataclass_fields


@ovld
def get_fields(instance: object):
    """Returns fields of an instance."""
    if is_dataclass(instance):
        return dataclass_fields(instance)
    else:
        msg = f"Expected dataclass"
        raise TypeError(msg)


__consolidate__ = True
__all__ = ["get_fields"]
