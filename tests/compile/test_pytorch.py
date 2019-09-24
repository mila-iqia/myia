# Most of the tests ar in test_backend, this is just for pytorch-specific
# tests that can't be made generic.
from myia.compile.backends import pytorch_default


def test_pytorch_defaults():
    d = pytorch_default(device='cuda')
    assert d['device'] == 'cuda:0'
    d = pytorch_default(device='cpu')
    assert d['device'] == 'cpu:0'
