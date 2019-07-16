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


from ..prim import ops as P
from ..composite import core


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
def tensor_dim(t):
    """Map of 'dim' pytorch method."""
    return len(P.shape(t))


@core
def t(a):
    """Map of 't' pytorch method."""
    return P.transpose(a, (1, 0))
