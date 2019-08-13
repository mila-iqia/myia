
from myia.abstract import AbstractValue, from_value
from myia.pipeline import (
    PipelineDefinition,
    scalar_pipeline,
    standard_pipeline,
)
from myia.utils import Merge

from . import steps


class Not:
    def __init__(self, value):
        self.value = value


class Options:

    def __init__(self, options):
        self.options = options

    def pipeline(self, default=steps.standard, config=None):
        if self.options['--scalar']:
            resources = scalar_pipeline.resources
        else:
            resources = standard_pipeline.resources
        all_steps = self.options['pipeline']
        pos = [p for p in all_steps if not isinstance(p, Not)]
        neg = {p.value for p in all_steps if isinstance(p, Not)}
        if not pos:
            pos = default
        final = [p for p in pos if p not in neg]
        pdef = PipelineDefinition(
            resources=resources,
            steps={p._name: p for p in final}
        )
        opts = self.options['opts']
        if opts and steps.opt not in final:
            raise Exception('Optimizations can only be applied if the'
                            ' opt step is in the pipeline')
        elif opts:
            pdef = pdef.configure({'opt.opts': Merge(opts)})

        if callable(config):
            pdef = config(pdef)
        elif config:
            pdef = pdef.configure(config)

        return pdef.make()

    def run(self, default=steps.standard, config=None):
        fn, = self.options['fns']
        pip = self.pipeline(default=default, config=config)
        argspec = self.argspec()
        return pip(input=fn, argspec=argspec)

    def argspec(self):
        return [
            v if isinstance(v, AbstractValue) else from_value(v, broaden=True)
            for v in self['args']
        ]

    def __getitem__(self, key):
        return self.options[key]
