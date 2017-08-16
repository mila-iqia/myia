"""
Simple Event and EventDispatcher.

These allow Myia's various components and transformations to
inform watchers about what they are doing. These watchers can
then identify problems, log actions, etc.
"""

from typing import Any, Union, Callable


Handler = Union['Event', Callable]


class Event(list):
    """
    Simple Event class.

    >>> e = Event('tickle')
    >>> e.register(lambda event, n: print("hi" * n))
    <function>
    >>> e(5)
    hihihihihi

    Event's constructor takes an additional argument, which
    is the EventDispatcher to which it belongs (or None).
    """
    def __init__(self,
                 name: str,
                 dispatcher: 'EventDispatcher' = None) -> None:
        self.name = name
        self.dispatcher = dispatcher
        self.owner: Any = dispatcher.owner if dispatcher else None

    def register(self, handler: Handler) -> Handler:
        # Returns the handler so it can be used as a decorator
        self.append(handler)
        return handler

    def __call__(self, *args, **kwargs) -> Any:
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
    """
    Handle arbitrary named events.

    * ``on_<event_name>(...)`` is shorthand for
      ``on(event_name, ...)``. Tip: it can be used as a
      decorator.
    * ``emit_<event_name>(...)`` is shorthand for
      ``emit(event_name, ...)``

    >>> d = EventDispatcher()
    >>> d.on_tickle(lambda event, n: print("hi" * n))
    <function>
    >>> d.emit_tickle(6)
    hihihihihihi

    An EventDispatcher also provides two special events:

    * ``NEW`` (all caps) is emitted whenever a new event
      is created (the first time ``on(event_name, ...)``
      is called for a given event_name).
    * ``ALL`` is triggered for all events.

    Tip: if you want to make event A fire when event B fires,
    you can write: ``d.on_B(d.emit_A)``.
    """

    def __init__(self, owner: Any = None, discover=True) -> None:
        self.owner = owner
        self._events = {'NEW': Event('NEW', self),
                        'ALL': Event('ALL', self)}
        if discover:
            discovery_event(owner or self, self)

    def on(self, event_name: str, *handlers: Callable) -> None:
        """
        Register one or more handlers
        """
        self[event_name].extend(handlers)

    def emit(self, event_name: str, *args, **kwargs) -> None:
        self[event_name](*args, **kwargs)

    def __getitem__(self, event_name: str) -> 'Event':
        e = self._events.get(event_name, None)
        if e is None:
            e = Event(event_name, self)
            self._events[event_name] = e
            self._events['NEW'](event_name, e)
            e.register(self._events['ALL'])
        return e

    def __getattr__(self, event_name: str) -> Union[Callable, Event]:
        if event_name.startswith('on_'):
            return self[event_name[3:]].register
        elif event_name.startswith('emit_'):
            return self[event_name[5:]]
        else:
            raise AttributeError(event_name)


# Newly constructed EventDispatchers emit this event so that
# we can hook to them more easily.
# Event signature: (discovery_event, dispatcher_owner, dispatcher)
discovery_event = Event('discovery')


def on_discovery(type: Any, event_name: str = None) -> Callable:
    """
    Mechanism to "discover" events. Given a type and an event name,
    ``on_discovery`` makes sure that your function triggers when
    that event is emitted by any instance of the given type created
    *after* the ``on_discovery`` call.

    The event name can be given explicitly, but if you decorate a
    function called ``on_<event_name>``, ``on_discovery`` will
    infer that this is the event you want to listen to.

    For example:

    >>> class MyClass(EventDispatcher): pass
    >>> @on_discovery(MyClass)
    ... def on_dog(*args):
    ...     print('I like doggies!')
    >>> c = MyClass()
    >>> c.emit_dog()
    I like doggies!
    """
    def decorate(fn: Callable) -> Callable:
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
