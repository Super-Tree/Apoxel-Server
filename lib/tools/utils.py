import numpy as np
import tensorflow as tf
from network.config import cfg

def fast_hist(labels, pred, num_class=2):
    k = (labels >= 0) & (labels < num_class)
    res = np.bincount(num_class * labels[k].astype(int) + pred[k], minlength=num_class**2).reshape(num_class, num_class)
    return res.astype(np.float32)

def scales_to_255(a, min_, max_, type_):
    return tf.cast(((a - min_) / float(max_ - min_)) * 255, dtype=type_)

def bounding_filter(points,box=(0,0)):

    x_min = box[0] - float(cfg.ANCHOR[0]) / 2
    x_max = box[0] + float(cfg.ANCHOR[0]) / 2
    y_min = box[1] - float(cfg.ANCHOR[1]) / 2
    y_max = box[1] + float(cfg.ANCHOR[1]) / 2
    z_min = cfg.Z_AXIS_MIN
    z_max = cfg.Z_AXIS_MAX

    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    f_filt = np.logical_and((x_points > x_min), (x_points < x_max))
    s_filt = np.logical_and((y_points > y_min), (y_points < y_max))
    z_filt = np.logical_and((z_points > z_min), (z_points < z_max))
    fliter = np.logical_and(np.logical_and(f_filt, s_filt), z_filt)
    indice = np.flatnonzero(fliter)
    filter_points = points[indice]

    return filter_points

def bound_trans_lidar2bv(points, center):
    points = bounding_filter(points)
    points = (points-center)*np.array([-1,-1,1])
    points = points[:,(0,1,2)]#TODO:be careful

    return points


def trans_bv2lidar(coord,center=(320,320)):
    coord = (coord-center)*np.array([-1,-1])
    coord = coord[:,(0,1)]#TODO:be careful

    return coord