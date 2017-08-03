
class Event(list):
    def __init__(self, name, dispatcher=None):
        self.name = name
        self.dispatcher = dispatcher
        self.owner = dispatcher.owner if dispatcher else None

    def register(self, handler):
        # Returns the handler so it can be used as a decorator
        self.append(handler)
        return handler

    def __call__(self, *args, **kwargs):
        if len(args) > 0 and not isinstance(args[0], Event):
            args = (self,) + args
        for f in self:
            f(*args, **kwargs)

    def __str__(self):
        return f'Event({self.name})'

    def __repr__(self):
        return str(self)

    def __hrepr__(self):
        return H.span['Event'](f'Event({self.name})')


class EventDispatcher:
    def __init__(self, owner=None, discover=True):
        self._events = {'NEW': Event('NEW', self),
                        'ALL': Event('ALL', self)}
        self.owner = owner
        if discover:
            discovery_event(owner or self, self)

    def on(self, event_name, *handlers):
        self[event_name].extend(handlers)

    def emit(self, event_name, *args, **kwargs):
        self[event_name](*args, **kwargs)

    def __getitem__(self, event_name):
        e = self._events.get(event_name, None)
        if e is None:
            e = Event(event_name, self)
            self._events[event_name] = e
            self._events['NEW'](event_name, e)
            e.register(self._events['ALL'])
        return e

    def __getattr__(self, event_name):
        if event_name.startswith('on_'):
            return self[event_name[3:]].register
        elif event_name.startswith('emit_'):
            return self[event_name[5:]]


# Newly constructed EventDispatchers emit this event so that
# we can hook to them more easily.
# Event signature: (discovery_event, dispatcher_owner, dispatcher)
discovery_event = Event('discovery')


def on_discovery(type, event_name=None):
    def decorate(fn):
        nonlocal event_name
        if event_name is None and fn.__name__.startswith('on_'):
            event_name = fn.__name__[3:]
        assert event_name is not None

        def seek(_, owner, dispatcher):
            if isinstance(owner, type):
                dispatcher.on(event_name, fn)
        discovery_event.register(seek)
        return fn
    return decorate
