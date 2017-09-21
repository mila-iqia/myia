
from bdb import Bdb
import pdb


class BucheDb(Bdb):
    __commands__ = [(x.split(':')[0], x.replace(':', '')) for x in [
        's:tep',
        'n:ext',
        'c:ontinue',
        'u:p',
        'd:own'
    ]]

    def __init__(self, buche, reader):
        super().__init__()
        self.buche = buche
        self.reader = reader
        self.frame = None
        self.display_frame = True

    def read(self):
        rval = self.reader.read()
        while rval.command != 'input':
            rval = self.reader.read()
        return rval

    def command_step(self):
        self.set_step()
        return True

    def command_next(self):
        self.set_next(self.get_frame())
        return True

    def command_continue(self):
        self.set_continue()
        return True

    def command_up(self):
        self.current = min(self.current + 1, len(self.frames) - 1)
        self.buche(self.get_frame())
        return False

    def command_down(self):
        self.current = max(self.current - 1, 0)
        self.buche(self.get_frame())
        return False

    def eval(self, code):
        frame = self.frames[self.current]
        gs = frame.f_globals
        ls = frame.f_locals

        if code.strip() == '':
            return

        self.buche.pre(code, kind='echo')

        lead = code.split(' ')[0]
        for begin, cmd in self.__commands__:
            if lead.startswith(begin) and cmd.startswith(lead):
                return getattr(self, f'command_{cmd}')()

        try:
            try:
                self.buche(eval(code, gs, ls), kind='result')
            except SyntaxError:
                exec(code, gs, ls)
        except Exception as exc:
            exc.__repl_string__ = code
            self.buche(exc, kind='error')
        return False

    def repl(self, frame):
        while True:
            cmd = self.read()
            if self.eval(cmd.contents):
                break

    def get_frame(self):
        return self.frames[self.current]

    def set_frame(self, frame, show=True, repl=True):
        self.current = 0
        self.frames = []
        fr = frame
        while fr:
            self.frames.append(fr)
            fr = fr.f_back
        if show and self.display_frame:
            self.buche(frame)
        if repl:
            self.repl(frame)

    def user_call(self, frame, args):
        self.set_frame(frame)

    def user_line(self, frame):
        self.set_frame(frame)

    # def user_return(self, frame, rval):
    #     self.set_frame(frame)

    def user_exception(self, frame, exc_info):
        self.buche('An exception occurred.')
        self.buche(exc_info)
        self.set_frame(frame)
