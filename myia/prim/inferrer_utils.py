"""Utilities that may leveraged by several inferrers."""


from ..infer import InferenceError, PartialInferrer, Context, ANYTHING, unwrap


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
        raise InferenceError('Item or attribute must be known.')

    try:
        mmap_t = mmap[type(data_t)]
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
            msg = f'Method {item_v} of {data_t} does not exist.'
            raise InferenceError(msg)

    else:
        # Module or static namespace
        data_v = await data['value']
        if data_v is ANYTHING:
            raise InferenceError('Data must be known.')
        if chk:
            chk(data_v, item_v)
        try:
            raw = fetch(data_v, item_v)
        except Exception as e:
            raise InferenceError('Getter error', e)
        value = resources.convert(raw)
        return track.from_value(value, Context.empty())
