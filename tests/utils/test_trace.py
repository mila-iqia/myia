
from collections import defaultdict
from itertools import count

from myia.utils import (
    DoTrace,
    Profiler,
    TraceExplorer,
    TraceListener,
    listener,
    resolve_tracers,
    tracer,
)


def day():
    with tracer('day'):
        with tracer('breakfast'):
            breakfast()
        with tracer('breakfast'):
            second_breakfast()
        with tracer('work'):
            work()
        with tracer('sleep'):
            sleep()


def breakfast():
    tracer().emit_fruit(fruit='banana', ripe=True)
    tracer().emit_fruit(fruit='apple')
    with tracer('cook', cookware='pan'):
        tracer().emit_eggs(method='sunny side up')
        tracer().emit_bacon(crispiness='crispy')


def second_breakfast():
    tracer().emit_crepe(sauce='chocolate')


def work(nhours=8):
    for i in range(nhours):
        with tracer(f'hour_{i + 1}') as tr:
            tracer().emit_play(game='solitaire')
        tr.set_results(productive=False)


def sleep():
    tracer().emit_dream(good=True, setting='home',
                        action='throwing cakes')
    tracer().emit_dream(good=False, setting='high school',
                        feeling='not this crap again')


def test_on():
    c = count()
    with TraceListener():
        tracer().on_fruit(lambda fruit, **kwargs: next(c))
        day()  # count 0 -> 2
    day()  # count 2 -> 2, because the listener does not outlast the with block
    assert str(c) == 'count(2)'


def test_explore():
    expected = {
        '/fruit': {'ripe': {bool}, 'fruit': {str}},
        '/cook/enter': {'cookware': {str}},
        '/cook/exit': {},
        '/cook/eggs': {'method': {str}},
        '/cook/bacon': {'crispiness': {str}},
    }
    with TraceExplorer() as explorer:
        breakfast()
    assert explorer.paths == expected


class _TraceCollector(TraceListener):
    def __init__(self, focus):
        super().__init__(focus)
        self.d = defaultdict(set)

    def collect(self, **kwargs):
        for k, v in kwargs.items():
            if not k.startswith('_'):
                self.d[k].add(v)

    def install(self, tracer):
        tracer.on(self.focus, self.collect)


def test_focus():

    with _TraceCollector('**') as tr:
        day()

    assert tr.d == {
        'fruit': {'apple', 'banana'},
        'ripe': {True},
        'cookware': {'pan'},
        'method': {'sunny side up'},
        'crispiness': {'crispy'},
        'sauce': {'chocolate'},
        'game': {'solitaire'},
        'good': {False, True},
        'setting': {'high school', 'home'},
        'action': {'throwing cakes'},
        'feeling': {'not this crap again'}
    }

    with _TraceCollector('fruit') as tr:
        day()

    assert tr.d == {
        'fruit': {'apple', 'banana'},
        'ripe': {True},
    }

    with _TraceCollector('sleep/**') as tr:
        day()

    assert tr.d == {
        'good': {False, True},
        'setting': {'high school', 'home'},
        'action': {'throwing cakes'},
        'feeling': {'not this crap again'}
    }


def test_profile():
    with Profiler():
        day()


def test_dotrace():
    handlers = {
        'fruit': lambda fruit, **kwargs: log.append(fruit),
        'sleep/**': lambda _event, **kwargs: log.append(_event)
    }

    log = []
    with DoTrace(handlers, post=lambda: log.append('end')):
        day()
    assert log == ['banana', 'apple', 'enter', 'dream', 'dream', 'exit', 'end']

    log = []
    with DoTrace(handlers, focus='breakfast/**/'):
        day()
    assert log == ['banana', 'apple']


def test_listener():
    log = []

    @listener('fruit', 'sleep/**')
    def do_log(_event, **kwargs):
        log.append(_event)

    with do_log():
        day()

    assert log == ['fruit', 'fruit', 'enter', 'dream', 'dream', 'exit']


def test_repr():
    with TraceListener():
        with tracer('xyz') as tr:
            assert str(tr) == '<TracerContextManager xyz>'


def test_resolve():
    from myia.utils import Named, Registry

    assert resolve_tracers('myia.utils.Named') == [(Named, ())]
    assert resolve_tracers('myia.utils.Named:BEEP') == [(Named, ["BEEP"])]
    assert (resolve_tracers('myia.utils.Named:BEEP:BOOP')
            == [(Named, ["BEEP", "BOOP"])])
    assert (resolve_tracers('myia.utils.Named("BEEP","BOOP")')
            == [(Named, ("BEEP", "BOOP"))])
    assert (resolve_tracers('myia.utils.Named;myia.utils.Registry')
            == [(Named, ()), (Registry, ())])
