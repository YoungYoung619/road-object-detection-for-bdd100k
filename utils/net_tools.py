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
        a feature map describes the offset from bbox to anchors
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


def decode_locations_one_layer(anchors_one_layer, offset_bboxes):
    """decode the offset bboxes into center bboxes
    Args:
        anchors_one_layer: ndarray represents all anchors coordinate in one layer,
                            encode by [y,x,h,w]
        offset_bboxes: A tensor with any shape ,the shape of lowest axis must be 4,
                            means the offset val in [y,x,h,w]
    Return:
        the locations of bboxes encode by [y,x,h,w]

    """
    shape = offset_bboxes.get_shape().as_list()
    try:
        i = shape.index(None)
        shape[i] = -1
    except ValueError:
        pass

    offset_bboxes = tf.reshape(offset_bboxes,shape=tf.stack([shape[0], -1, shape[-1]]))

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

    ## reshape to -1 ##
    anchor_cy = np.reshape(anchor_cy,[-1])
    anchor_cx = np.reshape(anchor_cx, [-1])
    anchor_h = np.reshape(anchor_h, [-1])
    anchor_w = np.reshape(anchor_w, [-1])


    bboxes_cy = offset_bboxes[:, :, 0] * anchor_h + anchor_cy
    bboxes_cx = offset_bboxes[:, :, 1] * anchor_w + anchor_cx
    bboxes_h = tf.exp(offset_bboxes[:, :, 2]) * anchor_h
    bboxes_w = tf.exp(offset_bboxes[:, :, 3]) * anchor_w

    cbboxes_out = tf.stack([bboxes_cy, bboxes_cx, bboxes_h, bboxes_w], axis=-1)
    cbboxes_out = tf.reshape(cbboxes_out, shape=shape)

    return cbboxes_out


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


def refine_groundtruth(anchors_all_layer, center_bboxes, labels, method, scope="refine_encode"):
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
        offset_val_ = encode_locations_one_layer(anchors_one_layer, center_bboxes[i])
        f = tf.expand_dims(offset_val_, axis=0)
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
                offset_val = encode_locations_one_layer(anchors_one_layer, center_bboxes[0])
                feat = tf.expand_dims(offset_val, axis=0)

                ## to get the which anchors are closest to the bboxes ##
                i,location_all_bboxes = tf.while_loop(get_t_info_condition, get_t_info,[i,feat],
                                        shape_invariants=[i.get_shape(),
                                                        tf.TensorShape([None]+feat.get_shape().as_list()[1:])])
                minDistance = tf.reduce_sum(location_all_bboxes*location_all_bboxes,axis=-1)
                minDistanceIndex = tf.argmin(minDistance,axis=0,output_type=tf.int32)
                # responsible_index.append(minDistanceIndex)

                ## init some vars to tf.while_loop ##
                gt_one_layer = tf.zeros(shape=offset_val.get_shape(),dtype=tf.float32)
                cbboxes_one_layer = tf.zeros(shape=offset_val.get_shape(), dtype=tf.float32)
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


def det_groundtruth(refine_out, offset_gt, cbboxes,
                refine_labels, refine_pos_mask, anchors, scope="det_encode"):
    """produce the ground truth for loss2
    Args:
        refine_out: a list which is composed by the output tensors from refine net, represents
                    the prediction of offset from obj to anchor.
        offset_gt: a list of tensor which is composed by the ground truth representing
            the offset from obj to initial anchor box in all layers
        refine_pos_mask: a list of tensor indicates which anchor is responsible for obj detection
        cbboxes: a list of tensor represents the bounding boxes encoded by [xc, yc, h, w].
        refine_labels: a list of tensor represents what kind of object in each initial anchor box.
        anchors: a list of ndarray indicates the initial anchor boxes position and size in all layers.
    Return:
        A list respresents the ground truth offset val for loss2 in all layers
        A list respresents the mask which means the anchor box's IOU > 0.5 in all layers
        A list represents the lable in all layers
    """
    mask_all_layers = []
    det_groundtruth = []
    det_labels = []
    pos_seg_config = config.det_pos_jac_val_all_layers
    with tf.name_scope(scope):
        for refine_out_one_l, offset_gt_one_l,cbboxes_one_l,\
            labels_one_l, refine_p_mask_one_l, anchors_one_l, pos_seg in zip(refine_out, offset_gt,
                                                                           cbboxes,refine_labels,refine_pos_mask,
                                                                           anchors, pos_seg_config):
            ## get the coordinate of anchor boxes after adjustment ##
            center_adanchors_one_layer = decode_locations_one_layer(anchors_one_l, refine_out_one_l)
            corner_adanchors_one_layer = centerBboxes_2_cornerBboxes(center_adanchors_one_layer)

            ## caculate the  jaccard of ground truth and anchor boxes after adjstment##
            corner_bboxes = centerBboxes_2_cornerBboxes(cbboxes_one_l)

            iou_one_layer = jaccard(corner_adanchors_one_layer, corner_bboxes)
            iou_one_layer_f = tf.expand_dims(iou_one_layer,axis=-1)
            positive_mask_oneL = tf.greater_equal(iou_one_layer_f, pos_seg)
            positive_mask_oneL = tf.cast(positive_mask_oneL, dtype=tf.int32)*tf.cast(refine_p_mask_one_l,tf.int32) ###filter some bboxes

            det_groundtruth.append((offset_gt_one_l-refine_out_one_l)*tf.cast(positive_mask_oneL,tf.float32))
            det_labels.append(labels_one_l*positive_mask_oneL)
            mask_all_layers.append(positive_mask_oneL)

    return det_groundtruth, mask_all_layers, det_labels


def smooth_l1(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.
    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


def refine_loss(refine_out, refine_groundtruth, refine_pos_mask, dtype=tf.float16):
    """get loss of refine net
    Args:
        refine_out: a list of refine out in all layers.
        refine_groundtruth: a list of refine ground-truth in all layers.
        refine_pos_mask: a list of mask indicate which anchor box resposible
                        for refine detection.
    Return: a Tensor represents the loss of refine net, with a shape (,).
    """
    ## process refine loss ##

    refine_pos_num_all_layers = []
    with tf.name_scope("refine_loss_process"):
        refine_loss = 0.
        i = 0
        for y, x, mask in zip(refine_groundtruth, refine_out, refine_pos_mask):
            bs = tf.cast(tf.shape(y)[0], dtype=dtype)
            mask = tf.cast(mask, dtype)
            y = tf.cast(y, dtype=dtype)
            refine_pos_num_one_layer = tf.reduce_sum(mask) / bs
            refine_pos_num_all_layers.append(refine_pos_num_one_layer)
            refine_loss_one_layer = tf.reduce_sum(smooth_l1((y - x) * mask)) / bs
            tf.summary.scalar("layer_%d_each_sample" % (i+1), refine_loss_one_layer / (refine_pos_num_one_layer+1e-5))
            refine_loss += refine_loss_one_layer
    return refine_loss


def det_clf_loss(refine_out, clf_out, det_out, det_groundtruth, det_pos_mask, det_labels, dtype=tf.float32):
    """build the loss of offset and classification
    Args:
        refine_out: the output of refine net, represents the prediciton of offset from initial anchor to groundtruth.
        clf_out: the output of clf net, represents the logits in each layers.
        det_out: the output of det out, represents the predicition of offset from refine anchor to groundtruth.
        det_groundtruth: represents the groundtruth of refine anchor to groundtruth.
        det_pos_mask: represnts the positive mask.
        det_labels: represnets the groundtruth of labels in each refine anchor.
    Return:
        detection_loss: represents the loss of offset from refine anchor to groundtruth
        classifition_loss: represents the loss of classification
    """
    bs = tf.cast(tf.shape(refine_out[0])[0], dtype=dtype)

    ## process det loss ##
    pos_num_all_layers = []
    with tf.name_scope("det_loss_process"):
        det_loss = 0.
        i = 0

        for y, x, m in zip(det_groundtruth, det_out, det_pos_mask):
            m = tf.cast(m, dtype=dtype)
            pos_num_one_layer = tf.reduce_sum(m)
            pos_num_all_layers.append(pos_num_one_layer / bs)
            tf.summary.scalar("layer_%d"%(i+1),pos_num_one_layer / bs)
            val = smooth_l1((y - x) * m)
            det_loss += tf.reduce_sum(val) / bs
            i += 1

    ## process clf loss ##
    with tf.name_scope("clf_loss_process"):
        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fpmask = []
        for labels, logits, m in zip(det_labels, clf_out, det_pos_mask):
            flogits.append(tf.reshape(logits, [-1, config.total_obj_n]))
            fgclasses.append(tf.reshape(labels, [-1]))
            fpmask.append(tf.reshape(m, [-1]))

        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        pmask = tf.cast(tf.concat(fpmask, axis=0), tf.bool)
        fpmask = tf.cast(pmask, dtype)

        n_positives = tf.reduce_sum(tf.cast(pmask, dtype))## positive num

        # Hard negative mining...
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)
        nmask = tf.logical_not(pmask)
        fnmask = tf.cast(nmask, dtype)

        nvalues = tf.where(nmask, predictions[:, 0], 1. - fnmask)

        # Number of negative entries to select.
        negative_ratio = 3.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + tf.cast(bs, dtype=tf.int32)
        n_neg = tf.minimum(n_neg, max_neg_entries)

        val, idxes = tf.nn.top_k(-nvalues, k=n_neg)
        max_hard_pred = -val[-1]
        tf.summary.scalar("max hard predition", max_hard_pred) ## the bigger, the better
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        with tf.name_scope('cross_entropy_pos'):
            clf_weights = np.array([1., 50., 4., 3., 8., 50., 20.,50., 1., 50., 50.])/280.
            gclasses = tf.one_hot(gclasses, config.total_obj_n)*clf_weights
            pos_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=gclasses)
            pos_loss = tf.div(tf.reduce_sum(pos_loss * fpmask), bs, name='value')

        with tf.name_scope('cross_entropy_neg'):
            neg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=no_classes)
            neg_loss = tf.div(tf.reduce_sum(neg_loss * fnmask), bs, name='value')

        clf_loss = neg_loss + pos_loss

    return det_loss, clf_loss


def optimizer(loss, global_step, learning_rate=1e-3, var_list=None):
    """build a optimizer to updata the weights
    Args:
        loss: a tensor reprensents the total loss
        global_step: a tensor represents the global step during training
    Return:
        a ops to updata the weights
    """
    # configure the learning rate#
    # learning_rate = tf.train.exponential_decay(self.__learningRate, self.__global_step_tensor,
    #                                            2 * self.__samplesNum / self.__batchSize, 0.94, staircase=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-8)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)

        ## clip the gradients ##
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var)
                        for grad, var in grads_and_vars]
        training_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

    return training_op


if __name__ == '__main__':
    # a = init_anchor(6)
    # b = init_anchor.anchor_info
    # a = n_anchor_each_layer('mobilenet_v2')

    a = anchors_one_layer((418, 418), (53, 53), init_anchor(6)['layer_1'])
    pass