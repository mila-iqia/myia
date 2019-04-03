
import os

if os.environ.get('BUCHE'):
    from debug import do_inject  # noqa
    from debug.butest import *  # noqa
