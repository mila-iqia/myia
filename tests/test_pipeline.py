
# import pytest

# from myia.pipeline import PipelineDefinition, PipelineStep, pipeline_function
# from myia.utils import Merge, Reset


# class OpStep(PipelineStep):

#     def __init__(self, pipeline_init, op, param=0):
#         super().__init__(pipeline_init)
#         self.op = op
#         self.param = param

#     def step(self, value):
#         return {'value': self.op(self.param, value)}


# class OpResourceStep(PipelineStep):

#     def __init__(self, pipeline_init, op):
#         super().__init__(pipeline_init)
#         self.op = op

#     def step(self, value):
#         return {'value': self.op(self.resources.param, value)}


# @pipeline_function
# def double_step(self, value):
#     return value * 2


# @pytest.fixture
# def op_pipeline():
#     return PipelineDefinition(
#         resources=dict(
#             param=2
#         ),
#         steps=dict(
#             addp=OpStep.partial(op=lambda p, x: p + x, param=1),
#             mulp=OpResourceStep.partial(op=lambda p, x: p * x),
#             neg=OpStep.partial(op=lambda p, x: -x),
#             square=OpStep.partial(op=lambda p, x: x * x),
#         )
#     )


# def test_PipelineDefinition_index(op_pipeline):
#     pdef = op_pipeline

#     assert pdef.index('addp') == 0
#     assert pdef.index('addp', True) == 1
#     assert pdef.index('!addp', True) == 0
#     assert pdef.index('mulp') == 1
#     assert pdef.index('neg') == 2
#     assert pdef.index('square') == 3

#     assert pdef.index(1) == 1

#     with pytest.raises(ValueError):
#         pdef.index('unknown')

#     with pytest.raises(TypeError):
#         pdef.index(object())


# def test_Pipeline(op_pipeline):
#     pdef = op_pipeline

#     pip = pdef.make()
#     assert pip(value=3) == {'value': 64}

#     assert pip.steps.addp(value=3) == {'value': 4}
#     assert pip.steps.mulp(value=3) == {'value': 8}
#     assert pip.steps.neg(value=3) == {'value': -8}
#     assert pip.steps.square(value=3) == {'value': 64}

#     assert pip['addp'](value=3) == {'value': 4}
#     assert pip['mulp'](value=3) == {'value': 8}
#     assert pip['neg'](value=3) == {'value': -8}
#     assert pip['square'](value=3) == {'value': 64}

#     assert pip['mulp':](value=3) == {'value': 36}
#     assert pip[:'mulp'](value=3) == {'value': 8}
#     assert pip[:-1](value=3) == {'value': -8}
#     assert pip['mulp':'neg'](value=3) == {'value': -6}

#     assert pip['!square'](value=3) == {'value': -8}
#     assert pip['mulp':'!neg'](value=3) == {'value': 6}

#     assert pdef['mulp':'neg'].run(value=3) == {'value': -6}


# def test_Pipeline_configure(op_pipeline):
#     pdef = op_pipeline

#     pip = pdef.configure(addp=Merge(param=2)).make()
#     assert pip(value=3) == {'value': 100}

#     pip = pdef.configure({'addp.param': 2}).make()
#     assert pip(value=3) == {'value': 100}

#     pip = pdef.configure(
#         {'addp.param': 2},
#         addp=Merge(op=lambda p, x: p - x)
#     ).make()
#     assert pip(value=3) == {'value': 4}

#     pip = pdef.configure(addp=Reset(op=lambda p, x: p - x, param=2)).make()
#     assert pip(value=3) == {'value': 4}

#     with pytest.raises(TypeError):
#         pdef.configure(addp=Reset(param=2)).make()

#     pip = pdef.configure(mulp=False).make()
#     assert pip(value=3) == {'value': 16}

#     pip = pdef.configure(mulp=False).configure(mulp=True).make()
#     assert pip(value=3) == {'value': 64}

#     pip = pdef.configure(mulp=False).configure(addp=False).make()
#     assert pip(value=3) == {'value': 9}

#     pip = pdef.configure(param=3).make()
#     assert pip(value=3) == {'value': 144}

#     with pytest.raises(KeyError):
#         pdef.configure(quack=[1, 2])

#     pdef2 = pdef.configure_resources(quack=[1, 2])
#     assert pdef2.make().resources.quack == [1, 2]
#     assert pdef2.configure(quack=Merge([3])).make().resources.quack \
#         == [1, 2, 3]
#     assert pdef2.configure(quack=[3]).make().resources.quack == [3]


# def test_Pipeline_insert(op_pipeline):
#     pdef = op_pipeline

#     half = OpStep.partial(op=lambda p, x: x / p, param=2)
#     double = double_step

#     pip = pdef.insert_before(half=half).make()
#     assert pip(value=3) == {'value': 25}

#     pip = pdef.insert_after(half=half).make()
#     assert pip(value=3) == {'value': 32}

#     pip = pdef.insert_before('addp', half=half).make()
#     assert pip(value=3) == {'value': 25}

#     pip = pdef.insert_after('addp', half=half).make()
#     assert pip(value=3) == {'value': 16}

#     pip = pdef.insert_before('square', half=half).make()
#     assert pip(value=3) == {'value': 16}

#     pip = pdef.insert_after('square', half=half).make()
#     assert pip(value=3) == {'value': 32}

#     pip = pdef.insert_after('square', double=double).make()
#     assert pip(value=3) == {'value': 128}


# def test_Pipeline_select(op_pipeline):
#     pdef = op_pipeline

#     pip = pdef.select('addp', 'neg').make()
#     assert pip(value=3) == {'value': -4}

#     pip = pdef.select('mulp', 'square').make()
#     assert pip(value=3) == {'value': 36}

#     pip = pdef.select('square', 'mulp').make()
#     assert pip(value=3) == {'value': 18}
