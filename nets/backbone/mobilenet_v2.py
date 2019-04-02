"""
Description :
Implementation of Mobilenet V2.
Architecture: https://arxiv.org/abs/1801.04381
Code: https://github.com/neuleaf/MobileNetV2
"""
import tensorflow as tf

import collections

weight_decay=1e-4
DTYPE = tf.float16

def relu(x, name='leaky_relu'):
    return tf.nn.leaky_relu(x, name=name)


def batch_norm(x, momentum=0.9, epsilon=1e-4, train=True, name='bn'):
    return tf.layers.batch_normalization(x,
                      momentum=momentum,
                      epsilon=epsilon,
                      scale=True,
                      training=train,
                      name=name)


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],dtype=DTYPE,
              regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        #tf.summary.histogram(w.name, w)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        if bias:
            biases = tf.get_variable('bias', [output_dim], dtype=DTYPE,
                                     initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def conv2d_block(input, out_dim, k, s, is_train, name):
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, out_dim, k, k, s, s, name='conv2d')
        net = batch_norm(net, train=is_train, name='bn')
        net = relu(net)
        return net


def conv_1x1(input, output_dim, name, bias=False):
    with tf.name_scope(name):
        return conv2d(input, output_dim, 1,1,1,1, stddev=0.02, name=name, bias=bias)

def pwise_block(input, output_dim, is_train, name, bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        out=conv_1x1(input, output_dim, bias=bias, name='pwb')
        out=batch_norm(out, train=is_train, name='bn')
        out=relu(out)
        return out


def dwise_conv(input, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1],
               padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
    with tf.variable_scope(name):
        in_channel=input.get_shape().as_list()[-1]
        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier], dtype=DTYPE,
                        regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None,name=None,data_format=None)
        if bias:
            biases = tf.get_variable('bias', [in_channel*channel_multiplier], dtype=DTYPE,
                                     initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        return conv


def res_block(input, expansion_ratio, output_dim, stride, is_train, name, bias=False, shortcut=True):
    with tf.name_scope(name), tf.variable_scope(name):
        # pw
        bottleneck_dim=round(expansion_ratio*input.get_shape().as_list()[-1])
        net = conv_1x1(input, bottleneck_dim, name='pw', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_bn')
        net = relu(net)
        # dw
        net = dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
        net = batch_norm(net, train=is_train, name='dw_bn')
        net = relu(net)
        # pw & linear
        net = conv_1x1(net, output_dim, name='pw_linear', bias=bias)
        net = batch_norm(net, train=is_train, name='pw_linear_bn')

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim=int(input.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins=conv_1x1(input, output_dim, name='ex_dim')
                net=ins+net
            else:
                net=input+net

        return net


def separable_conv(input, k_size, output_dim, stride, pad='SAME', channel_multiplier=1, name='sep_conv', bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]
        dwise_filter = tf.get_variable('dw', [k_size, k_size, in_channel, channel_multiplier], dtype=DTYPE,
                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                  initializer=tf.truncated_normal_initializer(stddev=0.02))

        pwise_filter = tf.get_variable('pw', [1, 1, in_channel*channel_multiplier, output_dim], dtype=DTYPE,
                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        strides = [1,stride, stride,1]

        conv=tf.nn.separable_conv2d(input,dwise_filter,pwise_filter,strides,padding=pad, name=name)
        if bias:
            biases = tf.get_variable('bias', [output_dim], dtype=DTYPE,
                                     initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv


def global_avg(x):
    with tf.name_scope('global_avg'):
        net=tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
        return net


def flatten(x):
    #flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    return tf.contrib.layers.flatten(x)


def pad2d(inputs, pad=(0, 0), mode='CONSTANT'):
    paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
    net = tf.pad(inputs, paddings, mode=mode)
    return net


def mobilenetv2(inputs, is_training=True):
    endPoints = collections.OrderedDict()
    exp = 6  # expansion ratio
    with tf.variable_scope('mobilenetv2'):
        net = conv2d_block(inputs, 32, 3, 2, is_training, name='conv1_1')  # size/2
        endPoints['layer_1'] = net
        net = res_block(net, 1, 16, 1, is_training, name='res2_1')
        endPoints['layer_2'] = net
        net = res_block(net, exp, 24, 2, is_training, name='res3_1')  # size/4
        endPoints['layer_3'] = net
        net = res_block(net, exp, 24, 1, is_training, name='res3_2')
        endPoints['layer_4'] = net

        net = res_block(net, exp, 32, 2, is_training, name='res4_1')  # size/8
        endPoints['layer_5'] = net
        net = res_block(net, exp, 32, 1, is_training, name='res4_2')
        endPoints['layer_6'] = net
        net = res_block(net, exp, 32, 1, is_training, name='res4_3')
        endPoints['layer_7'] = net

        net = res_block(net, exp, 64, 2, is_training, name='res5_1')   # size/16
        endPoints['layer_8'] = net
        net = res_block(net, exp, 64, 1, is_training, name='res5_2')
        endPoints['layer_9'] = net
        net = res_block(net, exp, 64, 1, is_training, name='res5_3')
        endPoints['layer_10'] = net
        net = res_block(net, exp, 64, 1, is_training, name='res5_4')
        endPoints['layer_11'] = net
        ##print(net)

        net = res_block(net, exp, 128, 1, is_training, name='res6_1')  # size/16
        endPoints['layer_12'] = net
        net = res_block(net, exp, 128, 1, is_training, name='res6_2')
        endPoints['layer_13'] = net
        net = res_block(net, exp, 128, 1, is_training, name='res6_3')
        endPoints['layer_14'] = net
        #print(net)

        net = res_block(net, exp, 256, 2, is_training, name='res7_1')  # size/32
        endPoints['layer_15'] = net
        net = res_block(net, exp, 256, 1, is_training, name='res7_2')
        endPoints['layer_16'] = net
        net = res_block(net, exp, 256, 1, is_training, name='res7_3')
        endPoints['layer_17'] = net
        #print(net)

        net = res_block(net, 2, 512, 2, is_training, name='res8_1')    # size/64
        endPoints['layer_18'] = net
        net = res_block(net, 2, 512, 1, is_training, name='res8_2')
        endPoints['layer_19'] = net

        net = res_block(net, 2, 256, 2, is_training, name='res9_1')       # size/128
        endPoints['layer_20'] = net
        net = res_block(net, 2, 256, 1, is_training, name='res9_2')
        endPoints['layer_21'] = net
        #print(net)

        net = res_block(net, 2, 256, 2, is_training, name='res10_1')  # size/256
        endPoints['layer_22'] = net
        net = res_block(net, 2, 256, 1, is_training, name='res10_2')
        endPoints['layer_23'] = net
        #print(net)

        # net = res_block(net, 3, 256, 2, is_training, name='conv11_1')  # size/128
        # endPoints['layer_24'] = net
        # net = res_block(net, 3, 256, 1, is_training, name='conv11_2')
        # endPoints['layer_25'] = net
        #print(net)
        return endPoints



import numpy as np

if __name__ == '__main__':
    imgs = tf.placeholder(dtype=DTYPE, shape=(None, 418, 418, 3))
    endpoints = mobilenetv2(inputs=imgs)

    print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    pass