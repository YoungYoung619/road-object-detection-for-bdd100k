"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
A net use for Catch-Detection

Author：Team Li
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

import utils.net_tools as net_tools
from nets.backbone.mobilenet.mobilenet_v2 import mobilenet_v2
from nets.backbone.mobilenet.mobilenet_v2 import training_scope
from nets.backbone.vgg import vgg_16
from nets.backbone.vgg import vgg_arg_scope
import config

slim = tf.contrib.slim

backbone_maps = {"mobilenet_v2":mobilenet_v2,
                 "vgg_16": vgg_16}
arg_maps = {"mobilenet_v2":training_scope,
                 "vgg_16": vgg_arg_scope}

class factory(object):
    """a class provide the net output"""
    def __init__(self, inputs, backbone_name, is_training, config_dict, dtype=tf.float32):
        """
        Args:
            inputs: imgs tensor with the shape [bs, h, w, c]
            backbone_name: a str indicates which backbone would be used.
            is_training:
            config: a dict indicate the config when building the whole net.
        """
        assert backbone_name in list(backbone_maps.keys())
        self. backbone_name = backbone_name

        ##build backbone
        with tf.variable_scope('backbone'):
            net = backbone_maps[backbone_name]
            with slim.arg_scope(arg_maps[backbone_name]()):
                end_points = net(inputs=inputs, is_training=is_training)

            backbone_feats = self.__process_backbone(end_points, config_dict['process_backbone_method'])
            self.backbone_feats = backbone_feats

        self.train_range = config_dict['train_range']

        ##build deconv and merge feats
        if config_dict['train_range'] is config.train_range.ALL:
            tf.logging.info('Building deconv net...')
            deconv_feats = self.__deconv_bone(backbone_feats, config_dict['deconv_method'],  is_training, dtype)
            self.deconv_feats  = deconv_feats
            tf.logging.info('Merging feats...')
            merge_feats = self.__merge_feats(backbone_feats, deconv_feats, config_dict['merge_method'])
            self.merge_feats = merge_feats

        ## build refine, det, clf net ##
        self.refine_out = self.__det_out(backbone_feats, is_training, scope='refine')

        if config_dict['train_range'] is config.train_range.ALL:
            self.clf_out = self.__clf_out(merge_feats, is_training, scope='clf')
            self.det_out = self.__det_out(merge_feats, is_training, scope='det')


    def __process_backbone(self, backbone_endpoints, method):
        """do some process in backbone endpoints
        Args:
            backbone_endpoints: endpoints of backbone
            method: the method of processing backbone endpoints
        Return:
            A dict: key---layer name
                    val---feature maps which would be used to train refine anchors
        """
        assert method in list(config.process_backbone_method.__members__.values())

        backbone_feats = collections.OrderedDict()

        if method is config.process_backbone_method.NONE:
            for index, feat_name in enumerate(config.extract_feat_name[self. backbone_name]):
                backbone_feats['layer_%s'%(str(index+1))] = backbone_endpoints[feat_name]
            return backbone_feats
        else:
            raise ValueError('Not support the method(%s)'%str(method))
            pass


    def __deconv_bone(self, backbone_feats, method, is_training, dtype, scope="deconv"):
        """build the upsample net
        Args:
            backbone_feats: a dict represents all the extracted feature maps from backbone
        Return:
            a list represents all the up-sample feature maps corresponding the input feat_layers
        """
        assert method in list(config.deconv_method.__members__.values())

        deconv_feats = collections.OrderedDict()
        feat_layers = list(backbone_feats.values())
        feat_layers.reverse()
        input = feat_layers[0]
        with tf.variable_scope(scope):
            for i in range(len(feat_layers)):
                with tf.variable_scope('block_%d' % (i + 1)):
                    if i == 0:
                        conv = slim.conv2d(input, input.get_shape().as_list()[-1], [1, 1], activation_fn=None)
                        input = slim.batch_norm(conv, is_training=is_training, activation_fn=tf.nn.leaky_relu)
                        deconv_feats['layer_%d' % (i + 1)] = input
                    else:
                        if method == config.deconv_method.LEARN_HALF:
                            f_c = int(feat_layers[i].get_shape().as_list()[-1]/2)
                            i_c = int(input.get_shape().as_list()[-1])

                            ## learn_half ##
                            # de_weight = tf.get_variable('dweight_%d' % (i), shape=[1, 1, f_c, i_c],
                            #                            regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                            #                            initializer=tf.truncated_normal_initializer(stddev=0.02))
                            de_weight = tf.get_variable('weight_%d' % (i), shape=[2, 2, f_c, i_c],
                                                        regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                                        initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                        dtype=dtype)
                            #tf.summary.histogram(de_weight.name, de_weight)
                            shape = feat_layers[i].get_shape().as_list()
                            # output_shape  = ops.convert_to_tensor([-1, shape[1], shape[2], f_c])
                            upsample = tf.nn.conv2d_transpose(input, de_weight, output_shape=[-1, shape[1], shape[2], f_c],
                                                              strides=[1, 2, 2, 1], padding="SAME")

                            # learn_half = slim.conv2d(upsample, f_c, [3, 3], activation_fn=None)
                            # learn_half = slim.batch_norm(learn_half, is_training=is_training, activation_fn=tf.nn.leaky_relu)
                            # learn_half = slim.conv2d(learn_half, f_c, [1, 1], activation_fn=None)
                            # learn_half = slim.batch_norm(learn_half, is_training=is_training, activation_fn=tf.nn.leaky_relu)

                            ## reuse half ##
                            # hw = learn_half.get_shape().as_list()[1:3]
                            hw = upsample.get_shape().as_list()[1:3]
                            reuse_half = tf.image.resize_images(input, hw)
                            reuse_half = tf.cast(reuse_half, dtype=dtype)
                            reuse_half = slim.conv2d(reuse_half, f_c, [1, 1], activation_fn=None)

                            input = tf.concat([upsample, reuse_half], axis=-1)
                            input = slim.batch_norm(input, is_training=is_training, activation_fn=tf.nn.leaky_relu)

                        elif method == config.deconv_method.LEARN_ALL:
                            f_c = int(feat_layers[i].get_shape().as_list()[-1])
                            i_c = int(input.get_shape().as_list()[-1])

                            # learn #
                            de_weight = tf.get_variable('weight_%d'%(i), shape=[2, 2, f_c, i_c],
                                                        regularizer=tf.contrib.layers.l2_regularizer(1e-4),
                                                        initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                        dtype=dtype)
                            #tf.summary.histogram(de_weight.name, de_weight)
                            upsample = tf.nn.conv2d_transpose(input,de_weight,output_shape=tf.shape(feat_layers[i]),
                                                              strides=[1,2,2,1],padding="SAME")
                            #upsample = slim.conv2d_transpose(input, channel, [1, 1], 2, activation_fn=None)
                            # learn_all = slim.conv2d(upsample, f_c, [3, 3], activation_fn=None)
                            # learn_all = slim.batch_norm(learn_all, is_training=is_training, activation_fn=tf.nn.leaky_relu)
                            # learn_all = slim.conv2d(learn_all, f_c, [1, 1], activation_fn=None)
                            input = slim.batch_norm(upsample, is_training=is_training, activation_fn=tf.nn.leaky_relu)
                        else:
                            raise ValueError('Parameter "method(%s)" wrong'%str(method))
                        deconv_feats['layer_%d'%(i+1)] = input
        return deconv_feats


    def __merge_feats(self, backbone_feats, deconv_feats, method, scope='merge'):
        """mix the features from backbone and deconvolution architecture
                Args:
                    down_endpoints: the dict-like extracted feature maps from backbone
                    up_endpoints: the dict-like extracted feature maps from deconvolution arichitecture
                    method: the feature merge method in net_config.py
                    is_training: indicates training or not
                Return:
                    return the merge feature endpoints
                """
        assert method in list(config.merge_method.__members__.values())

        merge_feats = collections.OrderedDict()
        backbone_feats = list(backbone_feats.values())
        deconv_feats = list(deconv_feats.values())
        deconv_feats.reverse()
        i = 1
        with tf.variable_scope(scope):
            for up_feat, de_feat in zip(backbone_feats, deconv_feats):
                with tf.variable_scope("block_%d" % (i)):
                    if method == config.merge_method.CONCAT:
                        merge_feats["layer_%d" % (i)] = tf.concat([up_feat, de_feat], axis=-1)
                    elif method == config.merge_method.ADD:
                        merge_feats["layer_%d" % (i)] = up_feat + de_feat
                    else:
                        raise ValueError('parameter "method(%s)" wrong...'%str(method))
                    i += 1
        return merge_feats


    def __det_out(self, feats, is_training, scope="detection"):
        """ produce the detection tensor
        Args:
            feats: the dict-like output of merge feature maps or backbone feature maps
            is_training:
        Return:
            a list of tensor represents the offset from initial(or refine)anchor to ground-truth
            the tensor has the shape with [bs, feat_h, feat_w, n_anchor_one_layer, 4]
        """
        ## get the number of anchors in each layer' cell ##
        n_anchor_each_layer = net_tools.n_anchor_each_layer(self.backbone_name)

        ## build refine or det net ##
        with tf.variable_scope(scope):
            det_conv_config = [[128, 4 * n_anchor_each_layer[0]], [128, 4 * n_anchor_each_layer[1]],
                               [128, 4 * n_anchor_each_layer[2]], [128, 4 * n_anchor_each_layer[3]],
                               [128, 4 * n_anchor_each_layer[4]], [128, 4 * n_anchor_each_layer[5]]]

            det_out = []
            for i, tensor in enumerate(list(feats.values())):
                input = tensor
                with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(1e-4)):
                    with tf.variable_scope("block_%d" % (i + 1)):
                        for channel in det_conv_config[i]:
                            ## learn half ##
                            conv = slim.conv2d(input, channel, [1, 1], activation_fn=None)
                            conv = slim.batch_norm(conv, is_training=is_training, activation_fn=tf.nn.leaky_relu)
                            conv = slim.conv2d(conv, channel, [3, 3], activation_fn=None)
                            output = slim.batch_norm(conv, is_training=is_training, activation_fn=tf.nn.leaky_relu)
                            input = output
                    hw = output.get_shape().as_list()[1:3]
                    output = tf.reshape(output, shape=[-1] + hw + [n_anchor_each_layer[i], 4])
                    det_out.append(output)
        return det_out


    def __clf_out(self, feats, is_training, scope='classification'):
        """ produce the logits for each layer.
        Args:
            feats: the dict-like output of merge feature maps
        Return:
            a list of tensor represents the class score,the tensor has the shape
            with [bs, feat_h, feat_w, anchor_num, obj_num]
        """

        ## get the number of anchors in each layer' cell ##
        n_anchor_each_layer = net_tools.n_anchor_each_layer(self.backbone_name)

        with tf.variable_scope(scope):
            clf_conv_config = [[128, config.total_obj_n * n_anchor_each_layer[0]], [128, config.total_obj_n * n_anchor_each_layer[1]],
                               [128, config.total_obj_n * n_anchor_each_layer[2]], [128, config.total_obj_n * n_anchor_each_layer[3]],
                               [128, config.total_obj_n * n_anchor_each_layer[4]], [128, config.total_obj_n * n_anchor_each_layer[5]]]
            clf_out = []
            for i, tensor in enumerate(list(feats.values())):
                input = tensor
                with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(1e-4)):
                    with tf.variable_scope("block_%d" % (i + 1)):
                        for channel in clf_conv_config[i]:
                            conv = slim.conv2d(input, channel, [1, 1], activation_fn=None)
                            conv = slim.batch_norm(conv, is_training=is_training, activation_fn=tf.nn.leaky_relu)
                            conv = slim.conv2d(conv, channel, [3, 3], activation_fn=None)
                            output = slim.batch_norm(conv, is_training=is_training, activation_fn=tf.nn.leaky_relu)
                            input = output
                    hw = output.get_shape().as_list()[1:3]
                    output = tf.reshape(output, shape=[-1] + hw + [n_anchor_each_layer[i], config.total_obj_n])
                    clf_out.append(output)  ##without softmax
        return clf_out

    def get_output(self):
        """get output
        Args:
            if train_range is REFINE, return refine_out
            if train_range is ALL, return refine_out, det_out, clf_out
        """
        if self.train_range is config.train_range.ALL:
            return self.refine_out, self.det_out, self.clf_out
        elif self.train_range is config.train_range.REFINE:
            return self.refine_out
        else:
            raise ValueError('Error')

if __name__ == '__main__':
    imgs = tf.placeholder(dtype=tf.float16, shape=[None, 418, 418, 3])
    config_dict = {'train_range':config.train_range.ALL,
                   'process_backbone_method': config.process_backbone_method.NONE,
                   'deconv_method':config.deconv_method.LEARN_HALF,
                   'merge_method':config.merge_method.ADD}
    net = factory(inputs=imgs, backbone_name='vgg_16', is_training=True, config_dict=config_dict)
    b = net.backbone_feats
    c = net.deconv_feats
    d = net.merge_feats
    refine_out, det_out, clf_out = net.get_output()
    pass