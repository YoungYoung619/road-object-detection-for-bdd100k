"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
from nets.catch_net import factory
from utils import net_tools
from utils import data_pileline_tools
from utils.common_tools import *
from utils.tf_extended import tf_utils
import config
from dataset import dataset_factory

from time import time
import os

import numpy as np
import cv2

import tensorflow as tf
slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'backbone_name', 'mobilenet_v2',
    'The name of the architecture to train.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'checkpoint_all', 'checkpoint/mobilenet_v2.model',
    'checkpoint(for all net) full name from which to fine-tune.')

tf.app.flags.DEFINE_integer(
    'vis_height', 720,
    'img height when visulization')

tf.app.flags.DEFINE_integer(
    'vis_width', 1080,
    'img height when visulization')

tf.app.flags.DEFINE_boolean(
    'vis_groundtruth', True,
    '')


FLAGS = tf.app.flags.FLAGS
DTYPE = tf.float32

global_step = tf.Variable(0, trainable=False, name='global_step')

def main(_):
    ## assert ##
    logger.info('Asserting parameters')
    assert FLAGS.checkpoint_all != None
    assert FLAGS.backbone_name in config.supported_backbone_name

    ## translate the anchor box config to x,y,h,w in all layers ##
    layer_n = len(list(config.extract_feat_name[FLAGS.backbone_name]))
    anchors_all = net_tools.anchors_all_layer(config.img_size,
                                         config.feat_size_all_layers[FLAGS.backbone_name],
                                        net_tools.init_anchor(layer_n))

    ## building data pileline ##
    logger.info('Building data pileline, using dataset---%s' % ('bdd100k_train'))
    with tf.device('/cpu:0'): ## use cpu to read data and batch data
        dataset = dataset_factory.get_dataset(
            'bdd100k', 'train', './dataset/bdd100k_TfRecord/')
        img, labels, bboxes = data_pileline_tools.prepare_data_test(dataset, num_readers=FLAGS.num_readers,
                                                                     batch_size=1, shuffle=True)

        center_bboxes = cornerBboxes_2_centerBboxes(bboxes)

        method = config.refine_method.JACCARD_BIGGER
        refine_gt, refine_cbboxes, refine_labels, refine_pos_mask = \
            net_tools.refine_groundtruth(anchors_all, center_bboxes, labels, method)

        list_shape = [1] + [layer_n] * 4
        batch_info_for_refine = tf.train.batch(
            tf_utils.reshape_list([img, refine_gt, refine_cbboxes, refine_labels, refine_pos_mask]),
            batch_size=1,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5)

        ## the batch img, gt for loss1, and responsible index ##
        imgs, refine_gt, refine_cbboxes, refine_labels, refine_pos_mask = \
            tf_utils.reshape_list(batch_info_for_refine, list_shape)

        norm_img = (2.0 / 255.0) * imgs - 1.0
        norm_img = tf.cast(norm_img, dtype=DTYPE)

    logger.info('Building model, using backbone---%s' % (FLAGS.backbone_name))
    config_dict = {'process_backbone_method': config.process_backbone_method.NONE,
                   'deconv_method': config.deconv_method.LEARN_HALF,
                   'merge_method': config.merge_method.ADD,
                   'train_range':config.train_range.ALL}
    net = factory(inputs=norm_img, backbone_name=FLAGS.backbone_name,
                  is_training=False, dtype=DTYPE, config_dict=config_dict)

    refine_out, det_out, clf_out = net.get_output()

    ## truth ##
    corner_bboxes_truth = []
    for box in refine_cbboxes:
        corner_bboxes_truth.append(tf.reshape(centerBboxes_2_cornerBboxes(box), [-1, 4]))
    corner_bboxes_gt = tf.concat(corner_bboxes_truth, axis=0)

    ## truth label ##
    labels = []
    for label_one_layer in refine_labels:
        label = tf.reshape(label_one_layer, [-1])
        labels.append(label)
    labels_gt = tf.concat(labels, axis=0)

    locations_all_layers = []  ##encode by [ymin, xmin, ymax, xmax]
    predition_all_layers = []

    for clf in clf_out:
        predition_all_layers.append(slim.softmax(clf))

    for refine_out, det_out, anchors_one_layer in \
            zip(refine_out, det_out, anchors_all):
        center_locations = net_tools.decode_locations_one_layer(anchors_one_layer, (refine_out+det_out))
        corner_locations = centerBboxes_2_cornerBboxes(center_locations)
        locations_all_layers.append(corner_locations)  ## [h,w,anchor_num,4]

    rscores, bboxes = net_tools.detected_bboxes(predition_all_layers, locations_all_layers, select_threshold=0.1,
                                           nms_threshold=0.4, top_k=400, keep_top_k=200)


    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # start queue
        coord = tf.train.Coordinator()
        # start the queues #
        threads = tf.train.start_queue_runners(coord=coord)

        tf.train.Saver().restore(sess, FLAGS.checkpoint_all)
        logger.info('Load checkpoint for all net success...')

        while True:
            bboxes_gt, Labels, bboxes_pred, scores_pred, img =\
            sess.run([corner_bboxes_gt, labels_gt, bboxes, rscores, imgs])

            img = img[0]
            img_gt = np.uint8(cv2.resize(img, dsize=(FLAGS.vis_width, FLAGS.vis_height)))
            img_pred = img_gt.copy()

            ## vis prediction
            for label, bboxes_np in bboxes_pred.items():
                scores = scores_pred[label]
                if scores.any() != 0.:
                    scores = scores[0]
                    bboxes_np = bboxes_np[0]
                    labels = np.full(dtype=np.int32, fill_value=label, shape=scores.shape)

                    img_pred = net_tools.visualize_boxes_and_labels_on_image_array(img_pred,
                                                                                   bboxes_np,
                                                                                   labels,
                                                                                   scores,
                                                                                   config.category_index,
                                                                                   skip_labels=True,
                                                                                   skip_scores=True)
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR)
            cv2.imshow('Prediction', img_pred)

            if FLAGS.vis_groundtruth:
                ##
                for bboxes_np, label in zip(bboxes_gt, Labels):
                    if label > 0:
                        bboxes_np = np.array([bboxes_np])
                        label = np.array([label])
                        img_gt = net_tools.visualize_boxes_and_labels_on_image_array(img_gt,
                                                                                       bboxes_np,
                                                                                       label,
                                                                                       np.array([1.]),
                                                                                       config.category_index,
                                                                                     skip_scores=True,
                                                                                     skip_labels=True)

                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)
                cv2.imshow('Ground-truth', img_gt)
            cv2.waitKey()
            cv2.destroyAllWindows()

            pass

        # terminate the threads #
        coord.request_stop()
        coord.join(threads)

    pass

if __name__ == '__main__':
    tf.app.run()