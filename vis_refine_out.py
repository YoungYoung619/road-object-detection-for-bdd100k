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
import config
from dataset import dataset_factory

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
    'checkpoint', 'checkpoint/mobilenet_v2.model',
    'checkpoint full name from which to fine-tune.')



#
# tf.app.flags.DEFINE_integer(
#     'save_summaries_secs', 60,
#     'The frequency with which summaries are saved, in seconds.')
#
# tf.app.flags.DEFINE_integer(
#     'save_interval_secs', 60,
#     'The frequency with which the model is saved, in seconds.')

# tf.app.flags.DEFINE_string(
#     'train_range', 'all',
#     'train all net or only for refine net')

FLAGS = tf.app.flags.FLAGS
DTYPE = tf.float32

global_step = tf.Variable(0, trainable=False, name='global_step')

def main(_):
    ## assert ##
    logger.info('Asserting parameters')
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
            capacity=5 * 1)

        ## the batch img, gt for loss1, and responsible index ##
        imgs, refine_gt, refine_cbboxes, refine_labels, refine_pos_mask = \
            tf_utils.reshape_list(batch_info_for_refine, list_shape)

        norm_imgs = (2.0 / 255.0) * imgs - 1.0
        norm_imgs = tf.cast(norm_imgs, dtype=DTYPE)

    logger.info('Building model, using backbone---%s' % (FLAGS.backbone_name))
    config_dict = {'train_range': config.train_range.REFINE,
                   'process_backbone_method': config.process_backbone_method.NONE,
                   'deconv_method': config.deconv_method.LEARN_HALF,
                   'merge_method': config.merge_method.ADD}
    net = factory(inputs=norm_imgs, backbone_name=FLAGS.backbone_name,
                  is_training=True, dtype=DTYPE, config_dict=config_dict)

    if config_dict['train_range'] is config.train_range.REFINE:
        ## get refine output ##
        refine_out = net.get_output()

        ## calculate the groudtruth of offset and classification ##
        det_gt, det_pos_mask, det_labels = \
            net_tools.det_groundtruth(refine_out, refine_gt, refine_cbboxes, refine_labels,
                                      refine_pos_mask, anchors_all)

        corner_bboxes_loss1 = []
        for box in refine_cbboxes:
            corner_bboxes_loss1.append(tf.reshape(centerBboxes_2_cornerBboxes(box), [-1, 4]))
        corner_bboxes_loss1 = tf.concat(corner_bboxes_loss1, axis=0)

        center_bboxes = []
        labels = []
        jac_mask = []
        pos_mask = []
        for refine_out,anchors_one_layer, label_one_layer, jac_mask_one_layer, pos_mask_one_layer \
                in zip(refine_out, anchors_all, refine_labels, refine_pos_mask, det_pos_mask):
            centerbboxes = net_tools.decode_locations_one_layer(anchors_one_layer,refine_out)
            centerbboxes = tf.reshape(centerbboxes, [-1,4])
            center_bboxes.append(centerbboxes)

            label = tf.reshape(label_one_layer, [-1])
            labels.append(label)

            jac_m = tf.reshape(jac_mask_one_layer, [-1])
            jac_mask.append(jac_m)

            pos_m = tf.reshape(pos_mask_one_layer, [-1])
            pos_mask.append(pos_m)

        center_bboxes = tf.concat(center_bboxes, axis=0)
        labels = tf.concat(labels, axis=0)
        jac_mask_flat = tf.concat(jac_mask, axis=0)
        pos_mask_flat = tf.concat(pos_mask, axis=0)

        center_bboxes_after_jac_mask = center_bboxes * tf.cast(tf.expand_dims(jac_mask_flat, axis=-1), dtype=tf.float32)
        corner_bboxes_after_jac_mask = centerBboxes_2_cornerBboxes(center_bboxes_after_jac_mask)

        center_bboxes_after_pos_mask = center_bboxes * tf.cast(tf.expand_dims(pos_mask_flat, axis=-1), dtype=tf.float32)
        corner_bboxes_after_pos_mask = centerBboxes_2_cornerBboxes(center_bboxes_after_pos_mask)

        pos_num = tf.reduce_sum(pos_mask_flat)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:

            if FLAGS.checkpoint == None:
                raise ValueError('checkpoint must not be none')
            else:
                tf.train.Saver().restore(sess, FLAGS.checkpoint)
                logger.info('Load checkpoint success...')

            # start queue
            coord = tf.train.Coordinator()
            # start the queues #
            threads = tf.train.start_queue_runners(coord=coord)

            while (True):
                IMG, CONER_JAC, CONER_POS, POSMASK, BBOXES_JAC, BBOXES_POS, CORNERBBOXES = \
                    sess.run([imgs, corner_bboxes_after_jac_mask, corner_bboxes_after_pos_mask, pos_mask_flat,
                              center_bboxes_after_jac_mask, center_bboxes_after_pos_mask, corner_bboxes_loss1])
                print("posnum is %d" % (np.sum(POSMASK)))

                showH = 512
                showW = 512

                IMG = np.uint8(IMG[0])
                IMG = cv2.resize(IMG, (showH, showW))
                rawImg = IMG.copy()
                IMG1 = IMG.copy()

                CONER_JAC[:, 0] *= showH
                CONER_JAC[:, 1] *= showW
                CONER_JAC[:, 2] *= showH
                CONER_JAC[:, 3] *= showW
                CONER_JAC = np.array(CONER_JAC, dtype=np.int16)

                CONER_POS[:, 0] *= showH
                CONER_POS[:, 1] *= showW
                CONER_POS[:, 2] *= showH
                CONER_POS[:, 3] *= showW
                CONER_POS = np.array(CONER_POS, dtype=np.int16)

                CORNERBBOXES[:, 0] *= showH
                CORNERBBOXES[:, 1] *= showW
                CORNERBBOXES[:, 2] *= showH
                CORNERBBOXES[:, 3] *= showW
                CORNERBBOXES = np.array(CORNERBBOXES, dtype=np.int16)

                #
                # PREDICT_POS[:, 0] *= self.__imgHeight
                # PREDICT_POS[:, 1] *= self.__imgWidth
                # PREDICT_POS[:, 2] *= self.__imgHeight
                # PREDICT_POS[:, 3] *= self.__imgWidth
                # PREDICT_POS = np.array(PREDICT_POS, dtype=np.int16)

                i = 0
                for bbox_jac in BBOXES_JAC:
                    if bbox_jac.any() != 0:
                        i += 1
                        cy = int(bbox_jac[0] * showH)
                        cx = int(bbox_jac[1] * showW)
                        cv2.circle(IMG, (cx, cy),
                                   radius=1, color=(0, 0, 255), thickness=1)

                for bbox_pos in BBOXES_POS:
                    if bbox_pos.any() != 0:
                        cy = int(bbox_pos[0] * showH)
                        cx = int(bbox_pos[1] * showW)
                        cv2.circle(IMG1, (cx, cy),
                                   radius=1, color=(0, 0, 255), thickness=1)

                for predict in CONER_POS:
                    if predict.any() != 0:
                        cv2.rectangle(IMG1, (predict[1], predict[0]), (predict[3], predict[2]), color=(0, 0, 255),
                                      thickness=1)

                for predict in CONER_JAC:
                    if predict.any() != 0:
                        cv2.rectangle(IMG, (predict[1], predict[0]), (predict[3], predict[2]), color=(0, 0, 255),
                                      thickness=1)

                for predict in CORNERBBOXES:
                    if predict.any() != 0:
                        cv2.rectangle(rawImg, (predict[1], predict[0]), (predict[3], predict[2]), color=(0, 0, 255),
                                      thickness=2)
                print(i)
                cv2.imshow("rawImg", rawImg)
                cv2.imshow("JAC", IMG)
                cv2.imshow("POS", IMG1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # terminate the threads #
            coord.request_stop()
            coord.join(threads)
    else:
        raise ValueError("error")
    pass

if __name__ == '__main__':
    tf.app.run()