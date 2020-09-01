import pytest

from myia.abstract import SHAPE, TYPE, AbstractArray
from myia.frontends import activate_frontend
from myia.testing.common import f64, to_abstract_test
from myia.validate import ValidationError, validate_abstract

activate_frontend("pytorch")
pytorch_abstract_types = pytest.importorskip(
    "myia_frontend_pytorch.pytorch_abstract_types"
)
PyTorchTensor = pytorch_abstract_types.PyTorchTensor


def test_validate_abstract_2():
    bad_array = AbstractArray(
        to_abstract_test(f64), {SHAPE: (1, 2), TYPE: PyTorchTensor}
    )
    with pytest.raises(ValidationError):
        validate_abstract(bad_array, uses={})
