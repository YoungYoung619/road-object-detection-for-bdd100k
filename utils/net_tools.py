"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
some general functions to train, predict, eval net.

Author：Team Li
"""
import config
import  math
import numpy as np
import collections

import tensorflow as tf
slim = tf.contrib.slim

from utils.common_tools import *
import utils.tf_extended as tfe

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


def refine_loss(refine_out, refine_groundtruth, refine_pos_mask, dtype=tf.float32):
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
    with tf.name_scope("det_clf_loss_process"):
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
                tf.summary.scalar("layer_%d_pos_num"%(i+1),pos_num_one_layer / bs)
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
                ## focal loss
                focal_factor = tf.square(tf.one_hot(gclasses, config.total_obj_n)*(1. - slim.softmax(logits)))
                pos_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=focal_factor)
                pos_loss = tf.div(tf.reduce_sum(pos_loss * fpmask), bs, name='value')

            with tf.name_scope('cross_entropy_neg'):
                neg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                              labels=no_classes)
                neg_loss = tf.div(tf.reduce_sum(neg_loss * fnmask), bs, name='value')

            clf_loss = neg_loss + pos_loss

        tf.summary.scalar('det_loss', det_loss)
        tf.summary.scalar('neg_loss', neg_loss)
        tf.summary.scalar('pos_loss', pos_loss)
        tf.summary.scalar('clf_loss', clf_loss)
        tf.summary.scalar('total_loss', clf_loss + det_loss)

        return det_loss, clf_loss


def optimizer(loss, global_step, batch_szie, learning_rate=1e-3, var_list=None, fix_learning_rate=True):
    """build a optimizer to updata the weights
    Args:
        loss: a tensor reprensents the total loss
        global_step: a tensor represents the global step during training
    Return:
        a ops to updata the weights
    """
    # configure the learning rate#
    if fix_learning_rate:
        l_rate = learning_rate
    else:
        l_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                70000 / batch_szie,
                                                0.97, staircase=True)
        tf.summary.scalar('learning_rate', l_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=l_rate, epsilon=1e-8)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)

        ## clip the gradients ##
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var)
                        for grad, var in grads_and_vars]
        training_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
        # training_op = optimizer.minimize(loss, global_step=global_step, var_list=var_list)

    return training_op


## nms ##
def bboxes_select_one_layer(predictions_layer, localizations_layer,
                            select_threshold=None, num_classes=21,
                            ignore_class=0, scope=None):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
        predictions_layer: A SSD prediction layer;
        localizations_layer: A SSD localization layer;
        select_threshold: Classification threshold for selecting a box. All boxes
            under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
        d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    select_threshold = 0.0 if select_threshold is None else select_threshold
    with tf.name_scope(scope, 'bboxes_select_layer',
                        [predictions_layer, localizations_layer]):
        # Reshape features: Batches x N x N_labels | 4
        p_shape = tfe.get_shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer,
                                        tf.stack([p_shape[0], -1, p_shape[-1]]))
        l_shape = tfe.get_shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer,
                                            tf.stack([l_shape[0], -1, l_shape[-1]]))

        d_scores = {}
        d_bboxes = {}
        for c in range(0, num_classes):
            if c != ignore_class:
                # Remove boxes under the threshold.
                scores = predictions_layer[:, :, c]
                fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
                scores = scores * fmask
                bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
                # Append to dictionary.
                d_scores[c] = scores
                d_bboxes[c] = bboxes

        return d_scores, d_bboxes


def bboxes_select_all_layers(predictions_net, localizations_net,
                             select_threshold=None,
                             num_classes=21,
                             ignore_class=0,
                             scope=None):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.
    Args:
        predictions_net: List of SSD prediction layers;
        localizations_net: List of localization layers;
        select_threshold: Classification threshold for selecting a box. All boxes
            under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
        d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    with tf.name_scope(scope, 'bboxes_select',
                           [predictions_net, localizations_net]):
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            scores, bboxes = bboxes_select_one_layer(predictions_net[i],
                                                     localizations_net[i],
                                                     select_threshold,
                                                     num_classes,
                                                     ignore_class)
            l_scores.append(scores)
            l_bboxes.append(bboxes)
        # Concat results.
        d_scores = {}
        d_bboxes = {}
        for c in l_scores[0].keys():
            ls = [s[c] for s in l_scores]
            lb = [b[c] for b in l_bboxes]
            d_scores[c] = tf.concat(ls, axis=1)
            d_bboxes[c] = tf.concat(lb, axis=1)
        return d_scores, d_bboxes


def detected_bboxes(predictions, localisations,
                    select_threshold=None, nms_threshold=0.5,
                    clipping_bbox=None, top_k=800, keep_top_k=200):
    """Get the detected bounding boxes from the SSD network output.
    """
    # Select top_k bboxes from predictions, and clip
    rscores, rbboxes = \
        bboxes_select_all_layers(predictions, localisations,
                                 select_threshold=select_threshold,
                                 num_classes=config.total_obj_n)
    rscores, rbboxes = \
        tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
    # Apply NMS algorithm.
    rscores, rbboxes = \
        tfe.bboxes_nms_batch(rscores, rbboxes,
                             nms_threshold=nms_threshold,
                             keep_top_k=keep_top_k)
    if clipping_bbox is not None:
        rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
    return rscores, rbboxes


########## for vis ############
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import collections

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).
  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.
  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))
  return image


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.
  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.
  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.
  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
  """Draws mask on an image.
  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
      values between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)
  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if np.any(np.logical_and(mask != 1, mask != 0)):
    raise ValueError('`mask` elements should be in [0, 1]')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))
  rgb = ImageColor.getrgb(color)
  pil_image = Image.fromarray(image)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))


def draw_keypoints_on_image(image,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True):
  """Draws keypoints on an image.
  Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  keypoints_x = [k[1] for k in keypoints]
  keypoints_y = [k[0] for k in keypoints]
  if use_normalized_coordinates:
    keypoints_x = tuple([im_width * x for x in keypoints_x])
    keypoints_y = tuple([im_height * y for y in keypoints_y])
  for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
    draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                  (keypoint_x + radius, keypoint_y + radius)],
                 outline=color, fill=color)


def draw_keypoints_on_image_array(image,
                                  keypoints,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True):
  """Draws keypoints on an image (numpy array).
  Args:
    image: a numpy array with shape [height, width, 3].
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_keypoints_on_image(image_pil, keypoints, color, radius,
                          use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    use_normalized_coordinates=True,
    max_boxes_to_draw=40,
    min_score_thresh=.2,
    agnostic_mode=False,
    line_thickness=3,
    groundtruth_box_visualization_color='red',
    skip_scores=False,
    skip_labels=False):
  """Overlay labeled boxes on an image with formatted scores and label names.
  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.
  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if instance_masks is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_masks_map[box],
          color=color
      )
    if instance_boundaries is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_boundaries_map[box],
          color='red',
          alpha=1.0
      )
    draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    if keypoints is not None:
      draw_keypoints_on_image_array(
          image,
          box_to_keypoints_map[box],
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates)

  return image
if __name__ == '__main__':
    # a = init_anchor(6)
    # b = init_anchor.anchor_info
    # a = n_anchor_each_layer('mobilenet_v2')

    a = anchors_one_layer((418, 418), (53, 53), init_anchor(6)['layer_1'])
    pass