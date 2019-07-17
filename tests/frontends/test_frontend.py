
import pytest


from myia.frontends import UnknownFrontend, activate_frontend


def test_load_frontend_unknown():
    with pytest.raises(UnknownFrontend):
        activate_frontend('_made_up_frontend')
