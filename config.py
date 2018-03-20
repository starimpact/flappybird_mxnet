# -*- coding: utf-8 -*-
# !/usr/bin/env python

# --------------------------------------------------------
# reinforce-deelp-learning-flappybird
# Copyright (c) 2016 SLXrobot
# Written by Chao Yu
# github : foolyc
# --------------------------------------------------------

import sys
#sys.path.insert(0, '/home/mingzhang/work/dmlc/python_mxnet/mxnet_v1.0.5/python')
sys.path.insert(0, '/home/mingzhang/work/dmlc/python_mxnet/python')
#sys.path.insert(0, '/home/mingzhang/work/dmlc/python_mxnet/python')
sys.path.append("game/")
#import mxnet as mx
#print 'version:', mx.__version__
#try:
#  print '#!#%!#$%!#$%   0'
#  from mxnet.base import _generate_op_module_signature
#  print '#!#%!#$%!#$%   1'
#  from mxnet.ndarray.register import _generate_ndarray_function_code
#  from mxnet.symbol.register import _generate_symbol_function_code
#  _generate_op_module_signature('mxnet', 'symbol', _generate_symbol_function_code)
#  _generate_op_module_signature('mxnet', 'ndarray', _generate_ndarray_function_code)
#except:
#  pass

FLG_GPU = True # using gpu or cpu
GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0000 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1 # number of frames to skip
FRAME = 4 # number of past frames to use as the input data of the q net
HEIGHT = 80 # height of input image
WIDTH = 80 # width of input image
UPDATE_STEP = 100 # target net updating period
SAVE_STEP = 10000 # saving the params per step period

