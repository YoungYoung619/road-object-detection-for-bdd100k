"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
some general functions

Author：Team Li
"""
import config
import  math
import numpy as np
import collections

import tensorflow as tf
slim = tf.contrib.slim

from utils.common_tools import *

def init_anchor(n_layers):
    """produce the initial anchor
    Args:
        n_layers: the number of feature layers which were used to detection.
    Return:
        A dict represents initial anchors in each layer, the val in anchors
        represents the height and width.
    """
    ## init the anchor box size after __extractFeatName be set##
    anchor_boxes = collections.OrderedDict()
    anchor_min = config.normal_anchor_range[0]
    anchor_max = config.normal_anchor_range[1]
    range_per_layer = (anchor_max - anchor_min) / (n_layers - 1)

    rangeMin = anchor_min
    rangeMax = rangeMin + range_per_layer
    h = config.img_size[0]
    w = config.img_size[1]
    for i in range(n_layers):
        if i == 0:
            f_1 = config.special_anchor_range[0]
            f_2 = config.special_anchor_range[1]
            anchor_one_layer = np.array([[f_1 * h, f_1 * w],
                                         [f_1 * h / math.sqrt(3),
                                          f_1 * w * math.sqrt(3)],
                                         [f_1 * h * math.sqrt(3),
                                          f_1 * w / math.sqrt(3)],
                                         [f_2 * h, f_2 * w],
                                         [f_2 * h / math.sqrt(3),
                                          f_2 * w * math.sqrt(3)],
                                         [f_2 * h * math.sqrt(3),
                                          f_2 * w / math.sqrt(3)]
                                         ])
        else:
            S_0 = rangeMin
            S_1 = (2 * rangeMin + rangeMax) / 3
            S_2 = (rangeMin + 2 * rangeMax) / 3
            anchor_one_layer = np.array([[S_0 * h, S_0 * w],
                                         [S_0 * h / math.sqrt(3),
                                          S_0 * w * math.sqrt(3)],
                                         [S_0 * h * math.sqrt(3),
                                          S_0 * w / math.sqrt(3)],
                                         [S_1 * h, S_1 * w],
                                         [S_1 * h / math.sqrt(3),
                                          S_1 * w * math.sqrt(3)],
                                         [S_1 * h * math.sqrt(3),
                                          S_1 * w / math.sqrt(3)],
                                         [S_2 * h, S_2 * w],
                                         [S_2 * h / math.sqrt(3),
                                          S_2 * w * math.sqrt(3)],
                                         [S_2 * h * math.sqrt(3),
                                          S_2 * w / math.sqrt(3)]
                                         ])
            rangeMin = rangeMax
            rangeMax = rangeMin + range_per_layer
            pass

        anchor_one_layer[:, 0] = np.minimum(anchor_one_layer[:, 0], h)
        anchor_one_layer[:, 1] = np.minimum(anchor_one_layer[:, 1], w)
        anchor_boxes["layer_%d" % (i + 1)] = anchor_one_layer

    return anchor_boxes

def n_anchor_each_layer(backbone_name):
    """
    Args:
        backbone_name: a str indicate backbone.
    Return:
        a list indicate the number of anchors in each leyer.
    """
    assert backbone_name in list(config.extract_feat_name.keys())
    anchor_boxes = init_anchor(len(config.extract_feat_name[backbone_name]))
    n_anchors = [hw_info.shape[0] for hw_info in list(anchor_boxes.values())]
    return n_anchors


######### anchor relative func #########
def anchors_one_layer(img_shape, feat_shape, anchors_one_layer, dtype=np.float32):
    """produce anchor coordinate in one layer
    Args:
        img_shape: a tuple or list describe img height and width
        feat_shape: a tuple or list, desribe the total num of anchor in hight
            direction and witdh direction , such as (38,38)----(heitht_size, width_size)
        anchors_one_layer: a ndarray desscribe the anchor size,
            shape is (num_anchor_one_layer,2), and 2 means height and width.

    Return: the x_center,y_center,height,width
        x_center is ndarray with a shape (heitht_size, width_size,1)
        y_center is ndarray with a shape (heitht_size, width_size,1)
        height is ndarray with a shape (num_anchor_one_layer)
        width is ndarray with a shape (num_anchor_one_layer)
    """
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    x_center = (x + 0.5)/ feat_shape[1]
    y_center = (y + 0.5)/ feat_shape[0]
    height = anchors_one_layer[:,0] / img_shape[0]
    width = anchors_one_layer[:,1] / img_shape[1]
    y_center = np.expand_dims(y_center,axis=-1)
    x_center = np.expand_dims(x_center, axis=-1)

    return y_center.astype(dtype), x_center.astype(dtype), \
           height.astype(dtype), width.astype(dtype)

def anchors_all_layer(img_shape, feats_shape, anchors_all_layer):
    """produce the coordinate of all anchors in all layer
    Args:
        img_shape: a tuple or list describe img height and width.
        feats_shape: a tuple or list, desribe the total num of anchors in all layers.
        anchors_all_layer: describe the anchors size in all layers.

    Return:
        A list describe the anchors coordinate in all layers.
    """
    anchors_all_layers = []
    for key, value in anchors_all_layer.items():
        y, x, h, w = anchors_one_layer(img_shape=img_shape,
                                            feat_shape=feats_shape[key],
                                            anchors_one_layer=value)
        anchors_all_layers.append([y,x,h,w])

    return anchors_all_layers
########################################


### loss relative func ###
def encode_locations_one_layer(anchors_one_layer, center_bbox):
    """ This function encodes a feature map relative to a designated bbox
    Args:
        anchors_one_layer: A list descibe the anchors coordinate in one layer.
        center_bbox: describe a bounding box coordinate ,should be [y, x, h, w]

    Return:
        a feature map describes the transition from bbox to anchors
    """
    yref, xref, href, wref = anchors_one_layer
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.

    anchor_ymin = np.float32(ymin)
    anchor_xmin = np.float32(xmin)
    anchor_ymax = np.float32(ymax)
    anchor_xmax = np.float32(xmax)

    # Transform to center / size.
    anchor_cy = (anchor_ymax + anchor_ymin) / 2.
    anchor_cx = (anchor_xmax + anchor_xmin) / 2.
    anchor_h = anchor_ymax - anchor_ymin
    anchor_w = anchor_xmax - anchor_xmin

    # Encode features.
    feat_cy = (center_bbox[0] - anchor_cy) / anchor_h
    feat_cx = (center_bbox[1] - anchor_cx) / anchor_w
    feat_h = tf.log(center_bbox[2] / anchor_h)
    feat_w = tf.log(center_bbox[3] / anchor_w)
    feat_location = tf.stack([feat_cy, feat_cx, feat_h, feat_w], axis=-1)
    return feat_location


def jaccard(anchors, corner_bbox):
    """Compute jaccard score between a box and the anchors.
    Args:
        anchors: A 2D tensor with shape(?,4), lowest axis means [ymin,xmin,ymax,xmax]
        corner_bbox: A 2D tensor with shape(?,4),means [ymin,xmin,ymax,xmax]
    Return:
        1D tensor respresents jaccard value.
    """
    shape = anchors.get_shape().as_list()
    try:
        i = shape.index(None)
        shape[i] = -1
    except ValueError:
        pass
    del shape[-1]
    anchors = tf.reshape(anchors, shape=[-1, 4])
    corner_bbox = tf.reshape(corner_bbox, shape=[-1, 4])
    vol_anchors = (anchors[:, 3] - anchors[:, 1]) * (anchors[:, 2] - anchors[:, 0])

    int_ymin = tf.maximum(anchors[:, 0], corner_bbox[:, 0])
    int_xmin = tf.maximum(anchors[:, 1], corner_bbox[:, 1])
    int_ymax = tf.minimum(anchors[:, 2], corner_bbox[:, 2])
    int_xmax = tf.minimum(anchors[:, 3], corner_bbox[:, 3])
    h = tf.maximum(int_ymax - int_ymin, 0.)
    w = tf.maximum(int_xmax - int_xmin, 0.)
    # Volumes.
    inter_vol = h * w
    union_vol = vol_anchors - inter_vol \
                + (corner_bbox[:, 2] - corner_bbox[:, 0]) * (corner_bbox[:, 3] - corner_bbox[:, 1])
    jaccard = (inter_vol / union_vol)
    return tf.reshape(jaccard, shape=shape)


def gt_refine_loss(anchors_all_layer, center_bboxes, labels, method, scope="refine_encode"):
    """produce ground truth for loss1
    Args:
        anchors_all_layer: anchor boxes coordinate in all layers.
        center_bboxes: all bounding boxes in one img encode type is [y,x,h,w],shape is (?,4)
        labels: all labels in one img, shape is (?)
        method：the gt config in gt_method

    Return:
        A list resprests the ground truth for all layers.
        A list resprests which anchor box is responsible for the center_bboxes.
    """
    #### for gt_method.NEAREST_NEIGHBOR #####
    def get_t_info_condition(i,feat):
        """Condition: check label index.
        """
        r = tf.less(i, tf.shape(center_bboxes[:,0]))
        return r[0]

    def get_t_info(i,feat):
        """
        """
        transition_val_ = encode_locations_one_layer(anchors_one_layer, center_bboxes[i])
        f = tf.expand_dims(transition_val_, axis=0)
        feat = tf.concat([feat,f],axis=0)
        return [i+1,feat]

    def condition_near(i,gt_one_layer,cbboxes_one_layer,labels_one_layer):
            """Condition: check label index.
            """
            r = tf.less(i, tf.shape(center_bboxes[:,0]))
            return r[0]

    def body_near(i,gt_one_layer,cbboxes_one_layer,labels_one_layer):
        index = tf.equal(minDistanceIndex, i)
        index = tf.cast(index, dtype=tf.float32)
        index = tf.expand_dims(index, axis=-1)

        cbboxes_one_layer += index*center_bboxes[i]
        gt_one_layer += index * encode_locations_one_layer(anchors_one_layer,center_bboxes[i])
        labels_one_layer += tf.cast(index,dtype=tf.int32)*tf.cast(labels[i],dtype=tf.int32)

        return [i+1,gt_one_layer,cbboxes_one_layer,labels_one_layer]
    #### for gt_method.NEAREST_NEIGHBOR #####

    ###### for gt_method.JACCARD_BIGGER #######
    def jaccard_bigger_condition_1(i,jac):
        """Condition: check label index.
        """
        r = tf.less(i, tf.shape(center_bboxes[:,0]))
        return r[0]

    def jaccard_bigger_body_1(i,jac):
        j = jaccard(corner_anchors_one_layer, centerBboxes_2_cornerBboxes(center_bboxes[i]))
        j = tf.expand_dims(j, axis=0)
        jac = tf.concat([jac, j], axis=0)
        return [i+1, jac]

    def jaccard_bigger_condition_2(i,gt_one_layer,cbboxes_one_layer,labels_one_layer):
        """Condition: check label index.
        """
        r = tf.less(i, tf.shape(center_bboxes[:,0]))
        return r[0]

    def jaccard_bigger_body_2(i,gt_one_layer,cbboxes_one_layer,labels_one_layer):
        mask = tf.equal(maxJacIndex, i)
        mask = tf.cast(mask, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        mask = mask * tf.cast(pos_mask,tf.float32)

        cbboxes_one_layer += mask*center_bboxes[i]
        gt_one_layer += mask*encode_locations_one_layer(anchors_one_layer,center_bboxes[i])
        labels_one_layer += tf.cast(mask,dtype=tf.int32)*tf.cast(labels[i],dtype=tf.int32)
        return [i+1,gt_one_layer,cbboxes_one_layer,labels_one_layer]
    ###### for gt_method.JACCARD_BIGGER #######

    gt_list = []
    ccboxes_list = []
    labels_list = []
    pos_mask_list = []
    pos_seg_config = config.refine_pos_jac_val_all_layers

    with tf.name_scope(scope):
        for anchors_one_layer, pos_val in zip(anchors_all_layer, pos_seg_config):
            if method == config.refine_method.NEAREST_NEIGHBOR:
                ## init some vars to tf.while_loop ##
                i = tf.constant(1)
                transition_val = encode_locations_one_layer(anchors_one_layer, center_bboxes[0])
                feat = tf.expand_dims(transition_val, axis=0)

                ## to get the which anchors are closest to the bboxes ##
                i,location_all_bboxes = tf.while_loop(get_t_info_condition, get_t_info,[i,feat],
                                        shape_invariants=[i.get_shape(),
                                                        tf.TensorShape([None]+feat.get_shape().as_list()[1:])])
                minDistance = tf.reduce_sum(location_all_bboxes*location_all_bboxes,axis=-1)
                minDistanceIndex = tf.argmin(minDistance,axis=0,output_type=tf.int32)
                # responsible_index.append(minDistanceIndex)

                ## init some vars to tf.while_loop ##
                gt_one_layer = tf.zeros(shape=transition_val.get_shape(),dtype=tf.float32)
                cbboxes_one_layer = tf.zeros(shape=transition_val.get_shape(), dtype=tf.float32)
                labels_one_layer = tf.zeros(shape=minDistanceIndex.get_shape().as_list()+[1], dtype=tf.int32)
                i = tf.constant(0)
                i,gt_one_layer,cbboxes_one_layer,labels_one_layer = tf.while_loop(condition_near, body_near,
                                                                                  [i, gt_one_layer,
                                                                                   cbboxes_one_layer,labels_one_layer])
                pos_mask = tf.ones(shape=minDistanceIndex.get_shape().as_list()+[1], dtype=tf.int32)
                pos_mask_list.append(pos_mask)
                gt_list.append(gt_one_layer)
                ccboxes_list.append(cbboxes_one_layer)
                labels_list.append(labels_one_layer)

            elif method == config.refine_method.JACCARD_BIGGER:
                bigger_val = pos_val

                yref, xref, href, wref = anchors_one_layer
                ymin = yref - href / 2.
                xmin = xref - wref / 2.
                ymax = yref + href / 2.
                xmax = xref + wref / 2.

                anchor_ymin = np.float32(ymin)
                anchor_xmin = np.float32(xmin)
                anchor_ymax = np.float32(ymax)
                anchor_xmax = np.float32(xmax)
                corner_anchors_one_layer = tf.stack([anchor_ymin,anchor_xmin,anchor_ymax,anchor_xmax], axis=-1)

                i = tf.constant(1)
                jac = jaccard(corner_anchors_one_layer,centerBboxes_2_cornerBboxes(center_bboxes[0]))
                jac = tf.expand_dims(jac,axis=0)
                i, jac_all_bboxes = tf.while_loop(jaccard_bigger_condition_1, jaccard_bigger_body_1, [i, jac],
                                                    shape_invariants=[i.get_shape(),
                                                                      tf.TensorShape([None] + jac.get_shape().as_list()[1:])])


                pos_mask = tf.reduce_max(jac_all_bboxes,axis=0)
                pos_mask = tf.greater_equal(pos_mask, bigger_val)
                pos_mask = tf.expand_dims(tf.cast(pos_mask, tf.int32),axis=-1)  ##filter the iou smaller than bigge_val
                maxJacIndex = tf.argmax(jac_all_bboxes, axis=0, output_type=tf.int32)

                gt_one_layer = tf.zeros(shape=maxJacIndex.get_shape().as_list()+[4], dtype=tf.float32)
                cbboxes_one_layer = tf.zeros(shape=maxJacIndex.get_shape().as_list()+[4], dtype=tf.float32)
                labels_one_layer = tf.zeros(shape=maxJacIndex.get_shape().as_list() + [1], dtype=tf.int32)
                i = tf.constant(0)
                i, gt_one_layer, cbboxes_one_layer, labels_one_layer =\
                    tf.while_loop(jaccard_bigger_condition_2, jaccard_bigger_body_2,[i, gt_one_layer,
                                                                                     cbboxes_one_layer,labels_one_layer])

                gt_list.append(gt_one_layer)
                ccboxes_list.append(cbboxes_one_layer)
                labels_list.append(labels_one_layer)
                pos_mask_list.append(pos_mask)

            elif method == config.refine_method.JACCARD_TOPK:
                raise ValueError('Not support now')
            else:
                raise ValueError('Function parameter "method" wrong')

        return gt_list, ccboxes_list, labels_list, pos_mask_list


if __name__ == '__main__':
    # a = init_anchor(6)
    # b = init_anchor.anchor_info
    # a = n_anchor_each_layer('mobilenet_v2')

    a = anchors_one_layer((418, 418), (53, 53), init_anchor(6)['layer_1'])
    pass