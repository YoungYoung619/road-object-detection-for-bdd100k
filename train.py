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

import tensorflow as tf
slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'backbone_name', 'mobilenet_v2',
    'The name of the architecture to train.')

tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

tf.app.flags.DEFINE_integer(
    'batch_size', 20, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'checkpoint_all', None,
    'checkpoint(for all net) full name from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_refine', 'checkpoint/mbn_none53x35/refine/mobilenet_v2.model',
    'checkpoint(for all net) full name from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'train_dir', 'checkpoint/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'summary_dir', 'summary/',
    'Directory where summary are written to.')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 20,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'summary_every_n_steps', 20,
    'The frequency with which summary are writed.')

tf.app.flags.DEFINE_integer(
    'save_every_n_steps', 2000,
    'The frequency with which model are saved.')

tf.app.flags.DEFINE_boolean(
    'fix_refine', True,
    'whether fix refine net')


FLAGS = tf.app.flags.FLAGS
DTYPE = tf.float32

global_step = tf.Variable(0, trainable=False, name='global_step')

def main(_):
    ## assert ##
    logger.info('Asserting parameters')
    assert FLAGS.batch_size > 0
    assert FLAGS.learning_rate >= 0.
    assert (FLAGS.log_every_n_steps > 0 or FLAGS.log_every_n_steps == None)
    assert (FLAGS.summary_every_n_steps > 0 or FLAGS.summary_every_n_steps == None)
    assert (FLAGS.save_every_n_steps > 0 or FLAGS.save_every_n_steps == None)
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
        img, labels, bboxes = data_pileline_tools.prepare_data_train(dataset, num_readers=FLAGS.num_readers,
                                                                     batch_size=FLAGS.batch_size, shuffle=True)

        center_bboxes = cornerBboxes_2_centerBboxes(bboxes)

        method = config.refine_method.JACCARD_BIGGER
        refine_gt, refine_cbboxes, refine_labels, refine_pos_mask = \
            net_tools.refine_groundtruth(anchors_all, center_bboxes, labels, method)

        list_shape = [1] + [layer_n] * 4
        batch_info_for_refine = tf.train.batch(
            tf_utils.reshape_list([img, refine_gt, refine_cbboxes, refine_labels, refine_pos_mask]),
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        ## the batch img, gt for loss1, and responsible index ##
        imgs, refine_gt, refine_cbboxes, refine_labels, refine_pos_mask = \
            tf_utils.reshape_list(batch_info_for_refine, list_shape)

        imgs = (2.0 / 255.0) * imgs - 1.0
        imgs = tf.cast(imgs, dtype=DTYPE)

    logger.info('Building model, using backbone---%s' % (FLAGS.backbone_name))
    config_dict = {'train_range': config.train_range.REFINE,
                   'process_backbone_method': config.process_backbone_method.NONE,
                   'deconv_method': config.deconv_method.LEARN_HALF,
                   'merge_method': config.merge_method.ADD}
    net = factory(inputs=imgs, backbone_name=FLAGS.backbone_name,
                  is_training=True, dtype=DTYPE, config_dict=config_dict)

    logger.info('Total trainable parameters:%s' %
                str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

    if config_dict['train_range'] is config.train_range.ALL:
        refine_out, det_out, clf_out = net.get_output()

        ## build refine loss ##
        refine_loss = net_tools.refine_loss(refine_out, refine_gt, refine_pos_mask, dtype=DTYPE)

        ## calculate the groudtruth of offset and classification ##
        det_gt, det_pos_mask, det_labels, iou_all_layers = \
            net_tools.det_groundtruth(refine_out, refine_gt, refine_cbboxes, refine_labels,
                                      refine_pos_mask, anchors_all)

        det_loss, clf_loss = net_tools.det_clf_loss(refine_out, clf_out, det_out, det_gt,
                                                    det_pos_mask, det_labels, iou_all_layers, dtype=DTYPE)

        ## reuse refine net ##
        reuse_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                       scope="backbone.+|refine.+")
        reuse_vars_dict = dict([(var.op.name, var) for var in reuse_vars])
        restore_saver = tf.train.Saver(reuse_vars_dict)

        if FLAGS.fix_refine:
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope="^((?!(backbone|refine)).)*$")  ##filter the refine model's vars
            total_loss = det_loss + clf_loss
        else:
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            total_loss = refine_loss + det_loss + clf_loss

        train_ops = net_tools.optimizer(total_loss, global_step, learning_rate=FLAGS.learning_rate,
                                        batch_szie=FLAGS.batch_size, var_list=train_vars,
                                        fix_learning_rate=False)

        ## merged the summary op and save the graph##
        summary_ops = tf.summary.merge_all()

        ## saver
        saver = tf.train.Saver(tf.global_variables())

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            ## create a summary writer ##
            writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

            if FLAGS.checkpoint_all == None:
                sess.run(tf.global_variables_initializer())
                logger.info('TF variables init success...')
            else:
                tf.train.Saver().restore(sess, FLAGS.checkpoint_all)
                logger.info('Load checkpoint for all net success...')

            if FLAGS.checkpoint_refine != None:
                restore_saver.restore(sess, FLAGS.checkpoint_refine)
                logger.info('Load checkpoint for refine net success...')

            # start queue
            coord = tf.train.Coordinator()
            # start the queues #
            threads = tf.train.start_queue_runners(coord=coord)

            avg_loss = 0.
            avg_clf_loss = 0.
            avg_det_loss = 0.
            avg_time = 0.

            tf.get_default_graph().finalize()

            while (True):
                start = time()
                update, summary, t_loss, c_loss, d_loss, current_step= \
                    sess.run([train_ops, summary_ops, total_loss, clf_loss, det_loss, global_step])
                t = round(time() - start, 3)

                ## for logging
                if FLAGS.log_every_n_steps != None:
                    ## caculate average loss ##
                    step = current_step % FLAGS.log_every_n_steps
                    avg_loss = (avg_loss * step + t_loss) / (step + 1.)
                    avg_clf_loss = (avg_clf_loss * step + c_loss) / (step + 1.)
                    avg_det_loss = (avg_det_loss * step + d_loss) / (step + 1.)
                    avg_time = (avg_time * step + t) / (step + 1.)

                    if current_step % FLAGS.log_every_n_steps == FLAGS.log_every_n_steps - 1:
                        ## print info ##
                        logger.info('Step%s total_loss:%s det_loss:%s clf_loss:%s time_each_step:%s' % \
                                    (str(current_step + 1), str(avg_loss), str(avg_det_loss), str(avg_clf_loss),
                                     str(avg_time)))
                        avg_loss = 0.
                        avg_clf_loss = 0.
                        avg_det_loss = 0.
                        avg_time = 0.

                ## for summary
                if FLAGS.summary_every_n_steps != None:
                    if current_step % FLAGS.summary_every_n_steps == FLAGS.summary_every_n_steps - 1:
                        writer.add_summary(summary, current_step)

                if FLAGS.save_every_n_steps != None:
                    if current_step % FLAGS.save_every_n_steps == FLAGS.save_every_n_steps - 1:
                        ## save model ##
                        logger.info('Saving model...')
                        model_name = os.path.join(FLAGS.train_dir, FLAGS.backbone_name + '.model')
                        saver.save(sess, model_name)
                        logger.info('Save model sucess...')

                if FLAGS.max_number_of_steps != None:
                    if current_step >= FLAGS.max_number_of_steps:
                        logger.info('Exit training...')
                        break

    elif config_dict['train_range'] is config.train_range.REFINE:
        ## get refine output ##
        refine_out = net.get_output()

        ## build refine loss ##
        refine_loss = net_tools.refine_loss(refine_out, refine_gt, refine_pos_mask, dtype=DTYPE)

        ## build optimizer ##
        train_ops = net_tools.optimizer(refine_loss, global_step, learning_rate=FLAGS.learning_rate,
                                        batch_szie=FLAGS.batch_size, fix_learning_rate=False)

        ## merged the summary op and save the graph##
        summary_ops = tf.summary.merge_all()

        # slim.learning.train(train_op=train_ops, logdir=FLAGS.train_dir, summary_op=merged,
        #                     number_of_steps=FLAGS.max_number_of_steps, log_every_n_steps=FLAGS.log_every_n_steps,
        #                     save_summaries_secs=FLAGS.save_summaries_secs, save_interval_secs=FLAGS.save_interval_secs)

        ## saver
        saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            ## create a summary writer ##
            writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

            if FLAGS.checkpoint_refine == None:
                sess.run(init)
                logger.info('TF variables init success...')
            else:
                tf.train.Saver().restore(sess, FLAGS.checkpoint_refine)
                logger.info('Load checkpoint success...')

            # start queue
            coord = tf.train.Coordinator()
            # start the queues #
            threads = tf.train.start_queue_runners(coord=coord)

            avg_refine_loss = 0.
            avg_time = 0.
            while (True):
                start = time()
                update, summary, r_loss, current_step = sess.run([train_ops, summary_ops, refine_loss, global_step])
                t = round(time() - start, 3)

                ## for logging
                if FLAGS.log_every_n_steps != None:
                    ## caculate average loss ##
                    step = current_step % FLAGS.log_every_n_steps
                    avg_refine_loss = (avg_refine_loss * step + r_loss) / (step + 1.)
                    avg_time = (avg_time * step + t) / (step + 1.)

                    if current_step % FLAGS.log_every_n_steps == FLAGS.log_every_n_steps - 1:
                        ## print info ##
                        logger.info('Step_%s refine_loss:%s time:%s' % (str(current_step+1),
                                                                        str(avg_refine_loss),
                                                                        str(avg_time)))
                        avg_refine_loss = 0.

                ## for summary
                if FLAGS.summary_every_n_steps != None:
                    if current_step % FLAGS.summary_every_n_steps == FLAGS.summary_every_n_steps - 1:
                        writer.add_summary(summary, current_step)

                if FLAGS.save_every_n_steps != None:
                    if current_step % FLAGS.save_every_n_steps == FLAGS.save_every_n_steps - 1:
                        ## save model ##
                        logger.info('Saving model...')
                        model_name = os.path.join(FLAGS.train_dir, FLAGS.backbone_name + '.model')
                        saver.save(sess, model_name)
                        logger.info('Save model sucess...')

                if FLAGS.max_number_of_steps != None:
                    if current_step >= FLAGS.max_number_of_steps:
                        logger.info('Exit training...')
                        break

            # terminate the threads #
            coord.request_stop()
            coord.join(threads)
    else:
        raise ValueError("error")
    pass

if __name__ == '__main__':
    tf.app.run()