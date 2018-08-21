"""Utilities that may leveraged by several inferrers."""


from ..dtype import Class, ismyiatype
from ..infer import InferenceError, PartialInferrer, Context, ANYTHING, unwrap


class MyiaNameError(InferenceError):
    """Raised when a name is not found in scope."""


class MyiaAttributeError(InferenceError):
    """Raised when an attribute is not found in a type or module."""


async def static_getter(track, data, item, fetch, chk=None):
    """Return an inferrer for resolve or getattr.

    Arguments:
        track: The track on which the inference operates.
        data: A ref to the data.
        item: A ref to the item/attribute.
        fetch: A function to resolve the item on the data.
        chk: A function to check the values inferred for the
            data and item.
    """
    resources = track.engine.pipeline.resources
    mmap = resources.method_map

    data_t = await data['type']
    item_v = await item['value']
    if item_v is ANYTHING:
        raise InferenceError(
            'The value of the attribute could not be inferred.'
        )

    if ismyiatype(data_t, Class):
        # Get field from Class
        if item_v in data_t.attributes:
            if track.name == 'type':
                return data_t.attributes[item_v]
            else:
                data_v = await data['value']
                if data_v is ANYTHING:
                    return track.default({'type': data_t})
                else:
                    return track.from_value(
                        getattr(data_v, item_v), Context.empty()
                    )
        elif item_v in data_t.methods:
            method = data_t.methods[item_v]
            method = track.engine.pipeline.resources.convert(method)
            inferrer = track.from_value(method, Context.empty())
            inferrer = unwrap(inferrer)
            return PartialInferrer(
                track,
                inferrer,
                [data]
            )
        else:
            raise InferenceError(f'Unknown field in {data_t}: {item_v}')

    # Try method map
    try:
        mmap_t = mmap[data_t]
    except KeyError:
        mmap_t = None

    if mmap_t is not None:
        # Method call
        if chk:
            chk(None, item_v)
        if item_v in mmap_t:
            method = mmap_t[item_v]
            method = track.engine.pipeline.resources.convert(method)
            inferrer = track.from_value(method, Context.empty())
            inferrer = unwrap(inferrer)
            return PartialInferrer(
                track,
                inferrer,
                [data]
            )
        else:
            msg = f"object of type {data_t} has no attribute '{item_v}'"
            raise MyiaAttributeError(msg)

    else:
        # Module or static namespace
        data_v = await data['value']
        if data_v is ANYTHING:
            raise InferenceError(
                'Could not infer the type or the value of the object'
                f" on which to resolve the attribute '{item_v}"
            )
        if chk:
            chk(data_v, item_v)
        try:
            raw = fetch(data_v, item_v)
        except NameError as e:
            raise MyiaNameError(f"Cannot resolve name '{item_v}'")
        except AttributeError as e:
            raise MyiaAttributeError(str(e))
        except Exception as e:  # pragma: no cover
            raise InferenceError(f'Unexpected error in getter: {e!r}')
        value = resources.convert(raw)
        return track.from_value(value, Context.empty())
