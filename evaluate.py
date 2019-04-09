"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
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
import utils.tf_extended as tfe
from dataset import dataset_factory
import math
import time

import tensorflow as tf
slim = tf.contrib.slim

from config import *

# =========================================================================== #
# model Flags.
# =========================================================================== #
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
    'checkpoint_path', 'checkpoint/',
    'checkpoint(for all net) full name from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'eval_dir', 'evaluation/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

# =========================================================================== #
# evaluation Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'select_threshold', 0.1, 'Selection threshold.')
tf.app.flags.DEFINE_integer(
    'select_top_k', 400, 'Select top-k detected bounding boxes.')
tf.app.flags.DEFINE_integer(
    'keep_top_k', 200, 'Keep top-k detected objects.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.4, 'Non-Maximum Selection threshold.')
tf.app.flags.DEFINE_float(
    'matching_threshold', 0.7, 'Matching threshold with groundtruth objects.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.8, 'GPU memory fraction to use.')


FLAGS = tf.app.flags.FLAGS
DTYPE = tf.float32

global_step = tf.Variable(0, trainable=False, name='global_step')

def flatten(x):
    result = []
    for el in x:
        if isinstance(el, tuple):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def main(_):
    ## assert ##
    logger.info('Asserting parameters')
    assert FLAGS.backbone_name in supported_backbone_name

    ## translate the anchor box config to x,y,h,w in all layers ##
    layer_n = len(list(extract_feat_name[FLAGS.backbone_name]))
    anchors_all = net_tools.anchors_all_layer(img_size,
                                         feat_size_all_layers[FLAGS.backbone_name],
                                        net_tools.init_anchor(layer_n))

    ## building data pileline ##
    logger.info('Building data pileline, using dataset---%s' % ('bdd100k_train'))
    with tf.device('/cpu:0'): ## use cpu to read data and batch data
        dataset = dataset_factory.get_dataset(
            'bdd100k', 'train', './dataset/bdd100k_TfRecord/')
        img, labels, bboxes = data_pileline_tools.prepare_data_test(dataset, num_readers=FLAGS.num_readers,
                                                                     batch_size=FLAGS.batch_size, shuffle=True)
        difficults = tf.zeros(tf.shape(labels), dtype=tf.int64)



        list_shape = [1] * 4
        batch_info_for_refine = tf.train.batch(
            tf_utils.reshape_list([img, labels, bboxes, difficults]),
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=FLAGS.batch_size,
            dynamic_pad=True)

        ## the batch img, gt for loss1, and responsible index ##
        b_imgs, b_glables, b_gbboxes, b_difficults = \
            tf_utils.reshape_list(batch_info_for_refine, list_shape)

        norm_img = (2.0 / 255.0) * b_imgs - 1.0
        norm_img = tf.cast(norm_img, dtype=DTYPE)

    logger.info('Building model, using backbone---%s' % (FLAGS.backbone_name))
    config_dict = {'process_backbone_method': process_backbone_method.NONE,
                   'deconv_method': deconv_method.LEARN_HALF,
                   'merge_method': merge_method.ADD,
                   'train_range':train_range.ALL}
    net = factory(inputs=norm_img, backbone_name=FLAGS.backbone_name,
                  is_training=False, dtype=DTYPE, config_dict=config_dict)

    refine_out, det_out, clf_out = net.get_output()

    locations_all_layers = []  ##encode by [ymin, xmin, ymax, xmax]
    predition_all_layers = []

    for clf in clf_out:
        predition_all_layers.append(slim.softmax(clf))

    for refine_out, det_out, anchors_one_layer in \
            zip(refine_out, det_out, anchors_all):
        center_locations = net_tools.decode_locations_one_layer(anchors_one_layer, (refine_out+det_out))
        corner_locations = centerBboxes_2_cornerBboxes(center_locations)
        locations_all_layers.append(corner_locations)  ## [h,w,anchor_num,4]

    # Performing post-processing on CPU: loop-intensive, usually more efficient.
    with tf.device('/device:CPU:0'):
        rscores, rbboxes = net_tools.detected_bboxes(predition_all_layers, locations_all_layers,
                                                    select_threshold=FLAGS.select_threshold,
                                                    nms_threshold=FLAGS.nms_threshold,
                                                    top_k=FLAGS.select_top_k,
                                                    keep_top_k=FLAGS.keep_top_k)

        # Compute TP and FP statistics.
        num_gbboxes, tp, fp, rscores = \
            tfe.bboxes_matching_batch(rscores.keys(), rscores, rbboxes,
                                      b_glables, b_gbboxes, b_difficults,
                                      matching_threshold=FLAGS.matching_threshold)

    variables_to_restore = slim.get_variables_to_restore()

    # =================================================================== #
    # Evaluation metrics.
    # =================================================================== #
    with tf.device('/device:CPU:0'):
        dict_metrics = {}

        # FP and TP metrics.
        tp_fp_metric = tfe.streaming_tp_fp_arrays(num_gbboxes, tp, fp, rscores)
        for c in tp_fp_metric[0].keys():
            dict_metrics['tp_fp_%s' % c] = (tp_fp_metric[0][c],
                                            tp_fp_metric[1][c])

        # Add to summaries precision/recall values.
        aps_voc07 = {}
        aps_voc12 = {}
        for c in tp_fp_metric[0].keys():
            # Precison and recall values.
            prec, rec = tfe.precision_recall(*tp_fp_metric[0][c])

            # Average precision VOC07.
            v = tfe.average_precision_voc07(prec, rec)
            summary_name = 'AP_VOC07/%s' % c
            op = tf.summary.scalar(summary_name, v, collections=[])
            # op = tf.Print(op, [v], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            aps_voc07[c] = v

            # Average precision VOC12.
            v = tfe.average_precision_voc12(prec, rec)
            summary_name = 'AP_VOC12/%s' % c
            op = tf.summary.scalar(summary_name, v, collections=[])
            # op = tf.Print(op, [v], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            aps_voc12[c] = v

        # Mean average precision VOC07.
        summary_name = 'AP_VOC07/mAP'
        mAP = tf.add_n(list(aps_voc07.values())) / len(aps_voc07)
        op = tf.summary.scalar(summary_name, mAP, collections=[])
        op = tf.Print(op, [mAP], summary_name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # Mean average precision VOC12.
        summary_name = 'AP_VOC12/mAP'
        mAP = tf.add_n(list(aps_voc12.values())) / len(aps_voc12)
        op = tf.summary.scalar(summary_name, mAP, collections=[])
        op = tf.Print(op, [mAP], summary_name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # Split into values and updates ops.
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(dict_metrics)

    # =================================================================== #
    # Evaluation loop.
    # =================================================================== #
    config_ = tf.ConfigProto()
    config_.gpu_options.allow_growth = True

    num_batches = math.ceil(3000 / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path
    tf.logging.info('Evaluating %s' % checkpoint_path)

    # Standard evaluation loop.
    start = time.time()
    slim.evaluation.evaluate_once(
        master='',
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=flatten(list(names_to_updates.values())),
        variables_to_restore=variables_to_restore,
        session_config=config_)
    # Log time spent.
    elapsed = time.time()
    elapsed = elapsed - start
    print('Time spent : %.3f seconds.' % elapsed)
    print('Time spent per BATCH: %.3f seconds.' % (elapsed / num_batches))


if __name__ == '__main__':
    tf.app.run()