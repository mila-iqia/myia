import os, sys 
sys.path.insert(0,'/Users/wln/project/myia')
from myia.composite import grad
from myia.prim import ops as P
from myia.prim.py_implementations import \
    array_map, bool_not, bool_eq, hastype, distribute, shape, \
    broadcast_shape, switch, identity, bool_and, typeof, scalar_cast, \
    scalar_add, scalar_exp, scalar_log, scalar_sin, scalar_cos, scalar_tan, \
    scalar_div, scalar_to_array, env_add
from myia.composite import hyper_add, zeros_like
from myia.api import myia
import numpy as np

from myia.debug.graph_vis import getGraphViz
from myia.pipeline import standard_pipeline, scalar_parse as parse
from myia.dtype import Bool, Int, UInt, Float, List, Array, Tuple, Function, \
    Object, pytype_to_myiatype

def getInferPip(fn):
    pip = standard_pipeline.select(
        'parse', 'resolve', 'infer'
    ).configure({
        'inferrer.erase_value': False,
        'inferrer.tracks.value.max_depth': 10
    }).make()
    fn = pip(input=fn, argspec=({'_erase_value': True, 'value': np.array([1.0,1.0]).astype(np.float32)},
                            {'_erase_value': True, 'value': np.array([1.0,1.0]).astype(np.float32)}))['graph']
    return fn

def getSpecializePip(fn):
    pip = standard_pipeline.select(
        'parse', 'resolve', 'infer', 'specialize'
    ).configure({
        'inferrer.erase_value': False,
        'inferrer.tracks.value.max_depth': 10
    }).make()
    fn = pip(input=fn, argspec=({'_erase_value': True, 'value': np.array(1.0).astype(np.float32)},
                            {'_erase_value': True, 'value': np.array(1.0).astype(np.float32)}))['graph']
    return fn

def getOptPip(fn):
    pip = standard_pipeline.select(
        'parse', 'resolve', 'infer', 'specialize', 'erase_class', 'opt', 'erase_tuple'
    ).configure({
        'inferrer.erase_value': False,
        'inferrer.tracks.value.max_depth': 10
    }).make()
    fn = pip(input=fn, argspec=({'_erase_value': True, 'value': np.array(1.0).astype(np.float32)},
                            {'_erase_value': True, 'value': np.array(1.0).astype(np.float32)}))['graph']
    return fn

def getInferPip(fn):
    pip = standard_pipeline.select(
        'parse', 'resolve', 'infer'
    ).configure({
        'inferrer.erase_value': False,
        'inferrer.tracks.value.max_depth': 10
    }).make()
    fn = pip(input=fn, argspec=({'_erase_value': True, 'value': np.array([1.0,1.0]).astype(np.float32)},
                            {'_erase_value': True, 'value': np.array([1.0,1.0]).astype(np.float32)}))['graph']
    return fn

def getSpecializePip(fn):
    pip = standard_pipeline.select(
        'parse', 'resolve', 'infer', 'specialize'
    ).configure({
        'inferrer.erase_value': False,
        'inferrer.tracks.value.max_depth': 10
    }).make()
    fn = pip(input=fn, argspec=({'_erase_value': True, 'value': np.array(1.0).astype(np.float32)},
                            {'_erase_value': True, 'value': np.array(1.0).astype(np.float32)}))['graph']
    return fn

def getOptPip(fn):
    pip = standard_pipeline.select(
        'parse', 'resolve', 'infer', 'specialize', 'erase_class', 'opt', 'erase_tuple'
    ).configure({
        'inferrer.erase_value': False,
        'inferrer.tracks.value.max_depth': 10
    }).make()
    fn = pip(input=fn, argspec=({'_erase_value': True, 'value': np.array(1.0).astype(np.float32)},
                            {'_erase_value': True, 'value': np.array(1.0).astype(np.float32)}))['graph']
    return fn

def f(x, y):
    a = x * y
    return a

@myia
def test_grad(x, y):
    dfdx = grad(f)
    return dfdx(x, y)

@getInferPip
def test_grad_infer(x, y):
    dfdx = grad(f)
    return dfdx(x, y)

@parse
def test_grad_parse(x, y):
    dfdx = grad(f)
    return dfdx(x, y)

@getSpecializePip
def test_grad_special(x, y):
    dfdx = grad(f)
    return dfdx(x, y)

@getOptPip
def test_grad_opt(x, y):
    dfdx = grad(f)
    return dfdx(x, y)

print(test_grad(1.0,2.0))

getGraphViz(test_grad_parse).render(filename='vis_out/test_grad_parse')
getGraphViz(test_grad_infer).render(filename='vis_out/test_grad_infer')
getGraphViz(test_grad_special).render(filename='vis_out/test_grad_special')
getGraphViz(test_grad_opt).render(filename='vis_out/test_grad_opt')