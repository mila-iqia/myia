import pytest

from myia.utils.pipeline import Pipeline


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
