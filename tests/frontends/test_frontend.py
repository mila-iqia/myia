
import pytest
from myia.frontends import (
    FrontendLoadingError,
    UnknownFrontend,
    activate_frontend,
)


def test_load_frontend_unknown():
    with pytest.raises(UnknownFrontend):
        activate_frontend('_made_up_frontend')


def test_frontend_error():
    from myia.frontends import _frontends
    name = '__testing_name000_'

    def f():
        raise ValueError('test')

    assert name not in _frontends
    _frontends[name] = f

    with pytest.raises(FrontendLoadingError):
        activate_frontend(name)

    del _frontends[name]
