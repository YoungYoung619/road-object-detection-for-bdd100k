# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from nets.backbone import custom_layers

slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def vgg_16(inputs,
           is_training=True,
           dropout_keep_prob=0.5,
           scope='vgg_16'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    scope: Optional scope for the variables.

  Returns:
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.batch_norm, slim.max_pool2d],
                        outputs_collections=end_points_collection,):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2],scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        ## additional layer for ssd ##
        with tf.variable_scope('block6'):
          net = slim.conv2d(net, 1024, [3, 3], activation_fn=tf.nn.relu, scope='conv6')
          # net = slim.batch_norm(net, is_training=is_training, activation_fn=tf.nn.relu, scope='conv6')
          net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)

        with tf.variable_scope('block7'):
          net = slim.conv2d(net, 1024, [1, 1], activation_fn=tf.nn.relu, scope='conv7')
          # net = slim.batch_norm(net, is_training=is_training, activation_fn=tf.nn.relu, scope='conv7')
          net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)

        with tf.variable_scope('block8'):
          net = slim.conv2d(net, 256, [1, 1], activation_fn=tf.nn.relu, scope='conv1x1')
          # net = slim.batch_norm(net, is_training=is_training, activation_fn=tf.nn.relu, scope='conv1x1')
          net = custom_layers.pad2d(net, pad=(1, 1))
          net = slim.conv2d(net, 512, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='conv3x3', padding='VALID')
          # net = slim.batch_norm(net, is_training=is_training, activation_fn=tf.nn.relu, scope='conv3x3')

        with tf.variable_scope('block9'):
          net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
          # net = slim.batch_norm(net, is_training=is_training, activation_fn=tf.nn.relu, scope='conv1x1')
          net = custom_layers.pad2d(net, pad=(1, 1))
          net = slim.conv2d(net, 256, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='conv3x3', padding='VALID')
          # net = slim.batch_norm(net, is_training=is_training, activation_fn=tf.nn.relu, scope='conv3x3')

        with tf.variable_scope('block10'):
          net = slim.conv2d(net, 128, [1, 1], activation_fn=tf.nn.relu, scope='conv1x1')
          # net = slim.batch_norm(net, is_training=is_training, activation_fn=tf.nn.relu, scope='conv1x1')
          #net = custom_layers.pad2d(net, pad=(1, 1))
          net = slim.conv2d(net, 256, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='conv3x3', padding='VALID')
          # net = slim.batch_norm(net, is_training=is_training, activation_fn=tf.nn.relu, scope='conv3x3')


        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        return end_points
vgg_16.default_image_size = 224

import numpy as np
if __name__ == '__main__':
    imgs = tf.placeholder(dtype=tf.float16, shape=(None, 418, 418, 3))
    endpoints = vgg_16(inputs=imgs, is_training=True)

    print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    pass
