
from .nodes import TupleNode
from .about import About


def transformer_method(transform, arg_index=1):
    def decorator(fn):
        def decorated(*args, **kw):
            with About(args[arg_index], transform):
                return fn(*args, **kw)
        return decorated
    return decorator


class Transformer:
    """
    Base class for Myia AST transformers.

    Upon transforming a node, ``Transformer.transform``
    transfers the original's location to the new node,
    if it doesn't have a location.

    Define methods called ``transform_<node_type>``,
    e.g. ``transform_Symbol``.
    """
    __transform__: str = None

    def transform(self, node, **kwargs):
        cls = node.__class__.__name__
        try:
            method = getattr(self, 'transform_' + cls)
        except AttributeError:
            raise Exception(
                "Unrecognized node type in {}: {}".format(
                    self.__class__.__name__, cls)
            )
        with About(node, self.__transform__):
            rval = method(node, **kwargs)
        return rval


# TODO: document
def maptup(fn, vals):
    if isinstance(vals, TupleNode):
        return TupleNode(maptup(fn, x) for x in vals.values)
    else:
        return fn(vals)


# TODO: document
def maptup2(fn, vals1, vals2):
    if isinstance(vals1, TupleNode):
        assert type(vals2) is tuple
        assert len(vals1.values) == len(vals2)
        return TupleNode(maptup2(fn, x, y)
                         for x, y in zip(vals1.values, vals2))
    else:
        return fn(vals1, vals2)
