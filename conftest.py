def myia_repr_failure(self, excinfo):
    exc = excinfo.value
    trace = getattr(exc, "myia_trace", None)
    # return self._repr_failure(excinfo)
    if trace is not None:
        from myia.ir.print import format_exc

        return format_exc(exc)
    else:
        return self._repr_failure(excinfo)


def pytest_collection_modifyitems(config, items):
    # Here we replace repr_failure on all the items so that they display a nice
    # error on InferenceError. This is very hacky but it works and I don't have
    # any more time to waste with pytest's nonsense.
    for item in items:
        typ = type(item)
        if not hasattr(typ, "_repr_failure"):
            typ._repr_failure = typ.repr_failure
            typ.repr_failure = myia_repr_failure
