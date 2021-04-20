from myia.utils.pipeline import LoopPipeline, Pipeline, partition_keywords


def f1(x, y):
    return x + y


def f2(x, y, **kwargs):
    return x * y


def test_partition_keywords():
    assert partition_keywords(f1, {"x": 1, "y": 2}) == ({"x": 1, "y": 2}, {})
    assert partition_keywords(f1, {"x": 1}) == ({"x": 1}, {})
    assert partition_keywords(f1, {"x": 1, "z": 2}) == ({"x": 1}, {"z": 2})

    assert partition_keywords(f2, {"x": 1, "y": 2}) == ({"x": 1, "y": 2}, {})
    assert partition_keywords(f2, {"x": 1, "z": 2}) == ({"x": 1, "z": 2}, {})


def step_double(value):
    return value * 2


def step_init(value):
    return {"original": value, "value": value}


def step_quad(value):
    return {"result": value * 4}


class MultiplyStep:
    def __init__(self, mul):
        self.mul = mul

    def __call__(self, value):
        return value * self.mul


def test_pipeline():
    pip = Pipeline(step_init, step_double)
    assert pip(value=3) == {"original": 3, "value": 6}


def test_pipeline2():
    pip = Pipeline(step_init, MultiplyStep(3))
    assert pip(value=3) == {"original": 3, "value": 9}


def test_pipeline_insert_after():
    pip = Pipeline(step_init).insert_after(step_init, step_double)
    assert pip(value=3) == {"original": 3, "value": 6}


def test_pipeline_without_step():
    pip = Pipeline(step_init, step_double).without_step(step_double)
    assert pip(value=3) == {"original": 3, "value": 3}

    pip = Pipeline(step_init, step_double).without_step(step_init)
    assert pip(value=3) == {"value": 6}


def test_make_transformer():
    pip = Pipeline(step_init, step_quad)
    tr = pip.make_transformer("value", "result")
    assert tr(3) == 12


def test_pipeline_default_arguments():
    pip = Pipeline(step_init, step_double, arguments={"value": 7})
    assert pip() == {"original": 7, "value": 14}
    assert pip(value=3) == {"original": 3, "value": 6}


def end_condition(value):
    return {"changes": value > 1}


def test_loop_pipeline():
    pip = LoopPipeline(Pipeline(MultiplyStep(0.5)), end_condition)
    assert pip(value=10) == {"value": 0.625, "changes": False}
    assert pip(value=0.75) == {"value": 0.375, "changes": False}
