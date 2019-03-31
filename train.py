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

import tensorflow as tf

tf.app.flags.DEFINE_string(
    'backbone_name', 'mobilenet_v2',
    'The name of the architecture to train.')

tf.app.flags.DEFINE_integer(
    'batch_size', 20, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

# tf.app.flags.DEFINE_string(
#     'train_range', 'all',
#     'train all net or only for refine net')

FLAGS = tf.app.flags.FLAGS
DTYPE = tf.float16

global_step = tf.Variable(0, trainable=False, name='global_step')

def main(_):
    ## translate the anchor box config to x,y,h,w in all layers ##
    layer_n = len(list(config.extract_feat_name[FLAGS.backbone_name]))
    anchors_all = net_tools.anchors_all_layer(config.img_size,
                                         config.feat_size_all_layers[FLAGS.backbone_name],
                                        net_tools.init_anchor(layer_n))

    ## create a dataset provider ##
    dataset = dataset_factory.get_dataset(
        'bdd100k', 'train', './dataset/bdd100k_TfRecord/')
    img, labels, bboxes = data_pileline_tools.prepare_data_train(dataset, num_readers=FLAGS.num_readers,
                                                                 batch_size=FLAGS.batch_size, shuffle=True)

    center_bboxes = cornerBboxes_2_centerBboxes(bboxes)

    method = config.refine_method.JACCARD_BIGGER
    refine_gt, refine_cbboxes, refine_labels, refine_pos_mask = \
        net_tools.gt_refine_loss(anchors_all, center_bboxes, labels, method)

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
    config_dict = {'train_range': config.train_range.REFINE,
                   'process_backbone_method': config.process_backbone_method.NONE,
                   'deconv_method': config.deconv_method.LEARN_HALF,
                   'merge_method': config.merge_method.ADD}
    net = factory(inputs=imgs, backbone_name='mobilenet_v2', is_training=True, config_dict=config_dict)

    if config_dict['train_range'] is config.train_range.ALL:
        refine_out, det_out, clf_out = net.get_output()
    elif config_dict['train_range'] is config.train_range.REFINE:
        refine_out = net.get_output()
    else:
        raise ValueError("error")

if __name__ == '__main__':
    tf.app.run()