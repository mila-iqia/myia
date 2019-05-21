
import os

if os.environ.get('BUCHE'):
    from debug import do_inject  # noqa
    from debug.butest import *  # noqa


def pytest_addoption(parser):
    parser.addoption('--gpu', action='store_true', dest="gpu",
                     default=False, help="enable gpu tests")


def pytest_configure(config):
    if not config.option.gpu:
        setattr(config.option, 'markexpr', 'not gpu')
