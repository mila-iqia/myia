"""PyTorch Frontend."""

#############################################################################
# WARNING:  None of this file is explicitly executed by pytest or forced    #
#           to be covered by Codecov. It's omitted in .coveragerc           #
#                                                                           #
#           It is instead only parsed by Myia.                              #
#                                                                           #
#           Output & Gradients of each function in this is/needs_to_be      #
#           compared with original pytorch function that function is        #
#           replacing via pytorch_*_map in /myia/frontends/pytorch.py       #
#                                                                           #
#           I.e. Every function in this should have a pytest test made      #
#           for it that asserts its Output & Gradients is equal to the      #
#           pytorch original.                                               #
#############################################################################

from .. import composite as C, dtype as D
from ..abstract import AbstractArray
from ..composite import core
from ..hypermap import hyper_map
from ..ir import MultitypeGraph
from ..prim import ops as P

# This import is for WIP
# from ..dtype import Bool, Int


# ############# THESE FUNCTIONS SHOULD BE IN ALPHABETICAL ORDER #############

# This is a helper function
@core
def _pair(x):
    if not P.hastype(x, D.TupleT):
        x = (x, x)
    return x


@core
def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1,
           groups=1):
    r"""Applies a Conv2d."""
    # noqa: D202
    """
    # This is for later versions of pytorch that support other paddings?
    if padding_mode != 'zeros':
        raise Exception("'zeros' is the only padding_mode that is currently
                        supported.")
    #"""

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    ret = P.conv2d(input, weight, stride, padding, dilation, groups)
    if bias is not None:
        ret = ret + bias.reshape((1, bias.shape[0], 1, 1))
    return ret


@core
def linear(input, weight, bias=None):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        ret = bias + input @ weight.t()
    else:
        output = input @ weight.t()
        if bias is not None:
            output = output + bias
        ret = output
    return ret


@core
def item(x):
    """Map of 'item' pytorch method."""
    return P.array_to_scalar(x.reshape(()))


@core
def relu(x):
    """Relu activation function."""
    return hyper_map(P.scalar_max, x, 0.0)


@core
def sigmoid(x):
    """Sigmoid activation function."""
    return (C.tanh(x / 2) + 1) / 2


''' # WIP
squeeze = MultitypeGraph('squeeze')


@squeeze.register(APT)
@core
def _squeeze(x):
    """Remove a dim (of length 1)."""
    raise NotImplementedError()


@squeeze.register(APT, Int)
@core
def _squeeze(x, d):
    """Remove a dim (of length 1)."""
    raise NotImplementedError()


softmax = MultitypeGraph('softmax')


@softmax.register(APT, Int)
@core
def _softmax(x, d):
    """Remove a dim (of length 1)."""
    raise NotImplementedError()


#def softmax(self: Tensor, dim: _int, dtype: _dtype) -> Tensor: ...
#TODO: how to specify that the 3rd argument of softmax is a dtype
@softmax.register(APT, Int, )
@core
def _softmax(x, d, dt):
    """Remove a dim (of length 1)."""
    raise NotImplementedError()
#'''

_sum = MultitypeGraph('_sum')


@_sum.register(AbstractArray)
@core
def __sum(x):

    return P.array_reduce(P.scalar_add, x, ())


''' # WIP
@_sum.register(APT, Int)
@core
def __sum(x, d):
    """Remove a dim (of length 1)."""
    raise Exception("NotImplementedError (in pytorch_functions.py)")

    orig_shp = x.values[SHAPE]

    """ # Hardcoded example of function
    array_squash = P.array_reduce(P.scalar_add, x, (2, 1))
    array_reduced = array_squash.reshape((2,))
    """

    """
    array_squash = P.array_reduce(
        P.scalar_add, x, orig_shp[:d]+(1,)+orig_shp[d+1:])
    array_reduced = array_squash.reshape(orig_shp[:d]+orig_shp[d+1:])
    #"""

    """
    orig_shp = list(orig_shp)
    orig_shp[d] = 1
    array_squash = P.array_reduce(P.scalar_add, x, tuple(orig_shp))
    del orig_shp[d]
    array_reduced = array_squash.reshape(tuple(orig_shp))
    #"""

    return array_reduced


@_sum.register(APT, Int, Bool)
@core
def __sum(x, d, kd):
    """Remove a dim (of length 1)."""
    raise NotImplementedError()
#'''
