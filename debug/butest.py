
import os
import json
import sys
import math
import pytest
from collections import defaultdict
from itertools import count
from dataclasses import dataclass
from buche import buche, H, Reader, Repl, CodeGlobals, BucheDb

from .gprint import mcss

template_path = f'{os.path.dirname(__file__)}/test_template.html'

_currid = count()
_sheet = None
_infos = {}
_capture = None
_reader = None
_code_globals = CodeGlobals()


def get_reader():
    global _reader
    if _reader is None:
        _reader = Reader(sys.stdin)
        _reader.start()
    return _reader


def decompose(item):
    filename = item.module.__name__
    if item.originalname:
        basetest = item.originalname
        variant = item.name[len(basetest):]
    else:
        basetest = item.name
        variant = 'main'
    return filename, basetest, variant


def idof(item):
    filename, basetest, variant = decompose(item)
    return _sheet.groups[filename][basetest][variant]


def actual_outcome(report):
    marks = report.item.own_markers
    xfail = any(m.name == 'xfail' for m in marks)
    outcome = report.outcome
    if outcome == 'passed' and xfail:
        outcome = 'xpassed'
    elif outcome == 'skipped' and xfail:
        outcome = 'xfailed'
    return outcome


def prout(out, bu):
    for line in out.split('\n'):
        try:
            cmd = json.loads(line)
            if 'parent' in cmd:
                cmd['parent'] = (bu.parent + cmd['parent']).rstrip('/')
            print(json.dumps(cmd))
        except ValueError:
            bu.html.code['stdout'](line)


@dataclass
class Information:
    item: "Item"  # noqa: F821
    report: "Report"  # noqa: F821
    interactor: "ReportInteractor"


class ReportSheet:

    def __init__(self, items):
        self.groups = defaultdict(lambda: defaultdict(dict))
        for item in items:
            filename, basetest, variant = decompose(item)
            self.groups[filename][basetest][variant] = f'test{next(_currid)}'

    def __hrepr__(self, H, hrepr):
        container = H.table()
        for filename, tests in self.groups.items():
            testgroups = []
            for testname, variants in tests.items():
                testboxes = []
                for variant, address in variants.items():
                    testbox = H.div['test-report'](
                        address=address,
                    )
                    testboxes.append(testbox)
                testgroups.append(H.div['testgroup-report'](*testboxes))
            entry = H.tr['testfile-report'](
                H.td(filename),
                H.td(*testgroups, 'X')
            )
            container = container(entry)

        return container


class TestResult:

    def __init__(self, report):
        self.report = report

    def __hrepr__(self, H, hrepr):
        if self.report.duration < 0.001:
            kind = 'short'
            width = 5
            height = 5
        else:
            kind = 'long'
            width = int(5 + ((math.log10(self.report.duration) + 4) * 2))
            height = width

        res = H.div['testresult',
                    f'testresult-{actual_outcome(self.report)}',
                    f'testresult-kind-{kind}'](
            style=f'width: {width}px; height: {height}px;'
                  f' border-radius: {width/2}px'
        )
        return res


class ReportInteractor:

    def __init__(self, id, item):
        self.id = id
        self.item = item
        self.report = None
        self.shown = False
        self.reported = False

    def set_report(self, report):
        self.report = report
        if self.shown:
            self.show_report()

    def show(self, synchronous=False):
        if self.shown:
            return

        self.repl = Repl(
            buche[f'main-tabs/tab-{self.id}'],
            get_reader(),
            code_globals=_code_globals
        )

        self.log = self.repl.log

        buche['main-tabs'].command_new(
            label=self.item.name,
            paneAddress=f'tab-{self.id}',
        )
        self.repl.start(synchronous=synchronous)
        self.shown = True

        if self.report is not None:
            self.show_report()

    def show_report(self):
        if self.reported:
            return

        self.log.html.b(f'Status: {actual_outcome(self.report)}')

        if self.report.excinfo:
            self.log(self.report.excinfo.value, interactive=True)

        if self.report.capstdout:
            self.log.html(H.div['report-stdout'](
                H.div('Captured stdout'),
                H.bucheLog(address='__stdout')
            ))
            prout(self.report.capstdout, self.log['__stdout'])

        if self.report.capstderr:
            self.log.html(H.div['report-stderr'](
                H.div('Captured stderr'),
                H.bucheLog(address='__stderr')
            ))
            prout(self.report.capstderr, self.log['__stderr'])

        self.reported = True


def pytest_sessionstart(session):
    global _capture
    _capture = session.config.pluginmanager.get_plugin('capturemanager')
    buche.command_template(src=template_path)
    buche(H.script(
        """
        function tippy(x, y) {
            f = () => tippy(x, y);
            setTimeout(f, 1000);
        }
        """,
        type="text/javascript"
    ))
    scripts = [
        "https://unpkg.com/popper.js@1/dist/umd/popper.min.js",
        "https://unpkg.com/tippy.js@4"
    ]
    for s in scripts:
        buche(H.script(type="text/javascript", src=s))
    buche(H.span())
    buche(H.style(mcss))
    buche.require('cytoscape')


def pytest_report_collectionfinish(config, startdir, items):
    global _sheet
    _sheet = ReportSheet(items)
    buche['main-tabs/pytest-reports'](_sheet)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    report.excinfo = call.excinfo
    report.item = item
    return report


def pytest_runtest_setup(item):
    id = idof(item)
    if id in _infos:
        raise Exception(id)
    interactor = ReportInteractor(id, item)
    _infos[id] = Information(item, None, interactor)

    class Db(BucheDb):

        def __init__(self):
            super().__init__(None)

        def set_trace(self, frame=None):
            interactor.show(synchronous=True)
            self.repl = interactor.repl
            super().set_trace(frame)

        def interaction(self, frame, tb):
            interactor.show(synchronous=True)
            self.repl = interactor.repl
            super().interaction(frame, tb)

    pytest.__pytestPDB._pdb_cls = Db


def pytest_report_teststatus(report):
    if report.when == 'call':
        # filename, basetest, variant = decompose(report.item)
        # id = _sheet.groups[filename][basetest][variant]
        id = idof(report.item)
        where = f'main-tabs/pytest-reports/{id}'

        print()
        buche[where](TestResult(report))
        expr = f"""
            tippy(this, {{
                content: `<table>
                <tr>
                    <td><b>Test</b></td>
                    <td>{report.item.name}</td>
                </tr>
                <tr>
                    <td><b>Outcome</b></td>
                    <td>{actual_outcome(report)}</td>
                </tr>
                <tr>
                    <td><b>Duration</b></td>
                    <td>{report.duration:.2f}s</td>
                </tr>
                </table>`
            }});
            this.onclick = function () {{
                this.bucheSend({{
                    path: this.buchePath || null,
                    eventType: "report",
                    id: this.getAttribute('address'),
                }});
            }};
        """

        buche[where].command_eval(expression=expr)
        _infos[id].interactor.set_report(report)

        return 'passed', '', ''
    else:
        return None


def pytest_sessionfinish(session, exitstatus):
    print()
    capture = session.config.pluginmanager.get_plugin('capturemanager')
    capture.stop_global_capturing()

    reader = get_reader()

    @reader.on_report
    def on_report(event, message):
        _infos[message.id].interactor.show()
