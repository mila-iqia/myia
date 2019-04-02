
import os

if os.environ.get('BUCHE'):
    from debug import do_inject
    from debug.butest import *
