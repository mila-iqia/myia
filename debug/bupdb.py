import os
import pdb

from buche import BucheDb

global_interactor = None


if os.environ.get("BUCHE"):
    class BuDb(BucheDb):
        def __init__(self):
            super().__init__(None)
            self.interactor = global_interactor

        def set_trace(self, frame=None):
            self.interactor.show(synchronous=True)
            self.repl = self.interactor.repl
            super().set_trace(frame)

        def interaction(self, frame, tb):
            self.interactor.show(synchronous=True)
            self.repl = self.interactor.repl
            super().interaction(frame, tb)

else:
    BuDb = pdb.Pdb
