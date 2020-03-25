"""Script to regenerate Python files in Myia.

This generates the following files:

* myia/operations/__init__.py
* myia/operations/primitives.py

"""

import importlib
import os

# Files to ignore in myia/operations
operations_ignore = ["utils.py", "primitives.py"]


# First lines of myia/operations/__init__.py
opinit_prelude = '''"""Myia operations."""

###############################################################################
# THIS FILE IS GENERATED AUTOMATICALLY. DO NOT EDIT!                          #
# To regenerate this file, run `python scripts/regen.py`                      #
# The script will search for all operations it can find in myia.operations    #
###############################################################################

from .utils import Operation, Primitive  # noqa'''


# First lines of myia/operations/primitives.py
prim_prelude = '''"""Primitive operations.

Primitive operations are handled as constants in the intermediate
representation, with the constant's value being an instance of a `Primitive`
subclass.

"""

###############################################################################
# THIS FILE IS GENERATED AUTOMATICALLY. DO NOT EDIT!                          #
# To regenerate this file, run `python scripts/regen.py`                      #
# The script will search for all primitives it can find in myia.operations    #
###############################################################################

from .utils import BackendPrimitive, InferencePrimitive, PlaceholderPrimitive'''  # noqa


# Format for an Operation
op_format = """
{registered_name} = Operation(
    name='{name}',
    defaults='{path}'
)"""


# Format for a Primitive
prim_format = """
{registered_name} = {primclass}(
    name='{name}',
    defaults='{path}'
)"""


def regen():
    """Regenerate all automatically generated Python files."""
    regen_operations()


def regen_operations():
    """Regenerate operations/__init__.py and operations/primitives.py."""
    oppaths = {}
    operations = {}
    primitives = {}

    def addop(module_name, data):
        data["path"] = module_name
        oppaths[module_name] = data
        operations[data["registered_name"]] = data

    def addprim(module_name, data):
        data["path"] = module_name
        primclass = {
            "backend": "BackendPrimitive",
            "inference": "InferencePrimitive",
            "placeholder": "PlaceholderPrimitive",
        }[data["type"]]
        data["primclass"] = primclass
        primitives[data["registered_name"]] = data

    for entry in os.listdir("myia/operations"):
        if entry in operations_ignore:
            continue
        if entry.startswith("_"):
            continue
        if not entry.endswith(".py"):
            continue

        module_name = f"myia.operations.{entry[:-3]}"
        mod = importlib.import_module(module_name)
        if hasattr(mod, "__operation_defaults__"):
            addop(module_name, mod.__operation_defaults__)
        if hasattr(mod, "__primitive_defaults__"):
            addprim(module_name, mod.__primitive_defaults__)

        for name, thing in vars(mod).items():
            if name.startswith("_"):
                continue
            path = f"{module_name}.{name}"
            if isinstance(thing, dict) and "registered_name" in thing:
                if "type" in thing:
                    addprim(path, thing)
                else:
                    addop(path, thing)

    oppath = "myia/operations/__init__.py"
    with open(oppath, "w") as opfile:
        print(opinit_prelude, file=opfile)
        for regname, data in sorted(operations.items()):
            print(op_format.format(**data), file=opfile)
    print(f"Generated {oppath}")

    primpath = "myia/operations/primitives.py"
    with open(primpath, "w") as primfile:
        print(prim_prelude, file=primfile)
        for regname, data in sorted(primitives.items()):
            print(prim_format.format(**data), file=primfile)
    print(f"Generated {primpath}")


if __name__ == "__main__":
    regen()
