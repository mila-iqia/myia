
import pytest

from myia import myia
from myia.debug import traceback  # noqa
from myia.frontends import UnknownFrontend, FrontendLoadingError, load_frontend


def test_load_frontend_unknown():
    with pytest.raises(UnknownFrontend):
        @myia(frontend='made_up_frontend')
        def step(x):
            return x


def test_frontend_error():
    from myia.frontends import _frontends, register_frontend
    name = '__testing_name000_'

    def f():
        raise ValueError('test')

    register_frontend(name, f)

    with pytest.raises(FrontendLoadingError):
        load_frontend(name)

    del _frontends[name]
