"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def centerBboxes_2_cornerBboxes(center_bboxes):
    """ change the center bounding boxes into corner bounding boxes
    Args:
        center_bboxes: a tensor, lowest dimention is [yc,xc,h,w]
    Return:
        a tensor with rank 4. lowest dimention means [ymin,xmin,ymax,xmax]
    """
    shape = center_bboxes.get_shape().as_list()
    try:
        i = shape.index(None)
        shape[i] = -1
    except ValueError:
        pass
    center_bboxes = tf.reshape(center_bboxes, shape=[-1, 4])
    ymin = center_bboxes[:, 0] - center_bboxes[:, 2] / 2  ##ymin = yc - h/2
    xmin = center_bboxes[:, 1] - center_bboxes[:, 3] / 2  ##xmin = xc - w/2
    ymax = center_bboxes[:, 0] + center_bboxes[:, 2] / 2  ##ymin = yc + h/2
    xmax = center_bboxes[:, 1] + center_bboxes[:, 3] / 2  ##xmin = xc - w/2
    corner_bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return tf.reshape(corner_bboxes, shape=shape)


def cornerBboxes_2_centerBboxes(corner_bboxes):
    """ change the center bounding boxes into corner bounding boxes
    Args:
        corner_bboxes: a tensor. lowest dimention means [ymin,xmin,ymax,xmax]
    Return:
        a tensor, has the same shape with input, lowest dimention means [yc,xc,h,w]
    """
    shape = corner_bboxes.get_shape().as_list()
    try:
        i = shape.index(None)
        shape[i] = -1
    except ValueError:
        pass
    corner_bboxes = tf.reshape(corner_bboxes, shape=[-1, 4])
    cy = (corner_bboxes[:, 0] + corner_bboxes[:, 2]) / 2.  ##yc = (ymin + ymax)/2
    cx = (corner_bboxes[:, 1] + corner_bboxes[:, 3]) / 2.  ##xc = (xmin + xmax)/2
    h = corner_bboxes[:, 2] - corner_bboxes[:, 0]  ##h = ymax - ymin
    w = corner_bboxes[:, 3] - corner_bboxes[:, 1]  ##w = xmax - xmin
    center_bboxes = tf.stack([cy, cx, h, w], axis=-1)
    return tf.reshape(center_bboxes, shape=shape)