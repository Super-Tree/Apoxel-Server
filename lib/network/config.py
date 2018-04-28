import numpy as np
import socket
import random
import string
import os.path as osp
from multiprocessing import cpu_count
from easydict import EasyDict as edict

__C = edict()

cfg = __C
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'RSdata32b'))
__C.RANDOM_STR = ''.join(random.sample(string.uppercase, 4))
__C.OUTPUT_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'output'))
__C.LOG_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'log'))
__C.TEST_RESULT = osp.abspath(osp.join(__C.ROOT_DIR, 'test_result'))

__C.CPU_CNT = cpu_count()

__C.VOXEL_POINT_COUNT = 50
__C.DETECTION_RANGE = 60.0
__C.Z_AXIS_MIN = -4.0
__C.Z_AXIS_MAX = 4.0
__C.ANCHOR = [__C.DETECTION_RANGE * 2, __C.DETECTION_RANGE * 2]
__C.CUBIC_RES = [0.18751, 0.18751]
__C.CUBIC_SIZE = [int(np.ceil(np.round(__C.ANCHOR[i] / __C.CUBIC_RES[i], 3))) for i in range(2)]

__C.TRAIN = edict()

__C.TRAIN.LEARNING_RATE = 1e-4

__C.TRAIN.ITER_DISPLAY = 10
__C.TRAIN.TENSORBOARD = True
__C.TRAIN.DEBUG_TIMELINE = True
__C.TRAIN.EPOCH_MODEL_SAVE = True
__C.TRAIN.USE_VALID = True
__C.TRAIN.FOCAL_LOSS = True
__C.TRAIN.VISUAL_VALID = True

__C.TEST = edict()
__C.TEST.DEBUG_TIMELINE = False
__C.TEST.ITER_DISPLAY = 1
