import pytest

from myia.pipeline import Pipeline, Resources
from myia.utils import Partializable


def step_double(value):
    return value * 2


def step_init(value):
    return {"original": value, "value": value}


def test_pipeline():
    pip = Pipeline(step_init, step_double)
    assert pip(value=3) == {"original": 3, "value": 6}

    pip = Pipeline(step_init).insert_after(step_init, step_double)
    assert pip(value=3) == {"original": 3, "value": 6}


def test_pipeline_default_arguments():
    pip = Pipeline(step_init, step_double, arguments={"value": 7})
    assert pip() == {"original": 7, "value": 14}
    assert pip(value=3) == {"original": 3, "value": 6}


class Shoe(Partializable):
    def __init__(self, model):
        self.model = model


def test_Resources():
    rdef = Resources.partial(
        quack=1, sandal=Shoe.partial(model="sandal")
    )
    rdef2 = rdef.configure({"sandal.model": "running"})

    r = rdef()
    r2 = rdef2()

    assert r.quack == 1
    assert r.sandal.model == "sandal"
    with pytest.raises(AttributeError):
        r.unknown

    assert r2.sandal.model == "running"
