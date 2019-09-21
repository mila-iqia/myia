"""Implementation of the 'resolve' operation."""

from ..lib import Constant, MyiaNameError, MyiaTypeError, Namespace, macro


@macro
async def resolve(info, r_data, r_item):
    """Perform static name resolution on a Namespace."""
    data_v, item_v = await info.build_all(r_data, r_item)
    if not isinstance(data_v, Namespace):  # pragma: no cover
        raise MyiaTypeError(
            f'data argument to resolve must be Namespace,'
            f' not {data_v}',
        )
    if not isinstance(item_v, str):  # pragma: no cover
        raise MyiaTypeError(
            f'item argument to resolve must be a string,'
            f' not {item_v}.',
        )
    try:
        resolved = data_v[item_v]
    except NameError:
        raise MyiaNameError(f"Cannot resolve name '{item_v}'")
    return Constant(resolved)


__operation_defaults__ = {
    'name': 'resolve',
    'registered_name': 'resolve',
    'mapping': resolve,
    'python_implementation': None,
}
