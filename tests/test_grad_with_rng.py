import numpy as np

from myia import myia, value_and_grad, xtype
from myia.operations import primitives as P, random_initialize, random_uint32
from myia.testing.multitest import bt


@bt()
def test_rstate_and_compute(backend):
    """Test grad in a function that uses and returns a rstate."""

    def rstate_and_compute(rstate, x):
        """Uses and returns a rstate with a computed value."""
        _rs, _val = random_uint32(rstate, (2, 3))
        _val = P.array_cast(_val, xtype.i64)
        return _rs, x * np.sum(_val).item()

    @myia(backend=backend)
    def step_init():
        """Myia function to initialize rstate."""
        return random_initialize(1234)

    @myia(backend=backend)
    def step_rstate_and_compute(rstate, x):
        """Compiled myia function that returns a rstate with computed value."""
        # Here it seems mandatory to use the `dout` parameters to get grad,
        # to help myia handle rstate grad correctly.
        (_rs, _val), _grad = value_and_grad(rstate_and_compute, "x")(
            rstate, x, dout=(1, 1)
        )
        return _rs, _val + _grad

    rstate = step_init()
    rstate, v1 = step_rstate_and_compute(rstate, 2)
    rstate, v2 = step_rstate_and_compute(rstate, 3)
    print(v1, v2)


@bt()
def test_only_compute(backend):
    """Test grad on a function that uses but does not return a rstate."""

    @myia(backend=backend)
    def step_init():
        """Myia function to initialize rstate."""
        return random_initialize(1234)

    def only_compute(rstate, x):
        """Use rstate but return only computed value."""
        _, _val = random_uint32(rstate, (2, 3))
        _val = P.array_cast(_val, xtype.i64)
        return x * np.sum(_val).item()

    @myia(backend=backend)
    def step_only_compute(rstate, x):
        """Compiled myia function that return only a computed value."""
        # Here dout seems not needed, as rstate is not returned.
        _val, _grad = value_and_grad(only_compute, "x")(rstate, x)
        return _val + _grad

    rstate = step_init()
    v1 = step_only_compute(rstate, 2)
    print(v1)
