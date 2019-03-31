"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import tensorflow as tf
slim = tf.contrib.slim

import config
from utils.augmentation import process
from utils.augmentation import tf_image

### train data pileline func###
def prepare_data_train(dataset, num_readers, batch_size, shuffle):
    """prepare batch data for training
    Args:
        dataset: should be a slim.dataset.Dataset
    Return:
        A tensor represents one img with specific height and width after augmentation,
        A tensor represents corresponding bboxes, value is in [0.，1.],
            shape is (?，4) means(ymin,xmin,ymax,xmax)
        A tensor represents corresponding labels, shape is (?)
    """
    # create a data provider #
    with tf.name_scope('data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(dataset, num_readers=num_readers,
                                                                  common_queue_capacity=5 * batch_size,
                                                                  common_queue_min=3 * batch_size,
                                                                  shuffle=shuffle)
    # Get for network: image, labels, bboxes.
    [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                     'object/label',
                                                     'object/bbox'])

    ## randomly crop and change the img ##
    img, labels, bboxes = process_raw_data_train(image=image, labels=glabels, bboxes=gbboxes,
                                                      out_shape=config.img_size)
    return img, labels, bboxes

def process_raw_data_train(image, labels, bboxes, out_shape):
    """process the raw img, raw glabels and raw gbboxes,
    and the raw img will be ramdonly crop and ramdonly
    change the light and color

    Args:
        image: A tensor representing an image of arbitrary size.
        labels: A tensor representing the obj label corresponding the input image.
        bboxes: A tensor represnting the bounding boxes corresponding the input image.(ymin,xmin,ymax,xmax)
        out_shape: means[output_height,output_width]
            output_height: The height of the image after preprocessing.
            output_wihdth: The width of the image after preprocessing.

    Returns:
        A preprocessed image.
    """
    # Distort image and bounding boxes.
    # dst_image = image
    dst_image, labels, bboxes, distort_bbox = \
        process.distorted_bounding_box_crop(image, labels, bboxes,
                                            min_object_covered=0.4,
                                            aspect_ratio_range=(0.6, 1.67))
    # Resize image to output size.
    dst_image = tf_image.resize_image(dst_image, out_shape,
                                      method=tf.image.ResizeMethod.BILINEAR,
                                      align_corners=False)

    # Randomly flip the image horizontally.
    dst_image, bboxes = tf_image.random_flip_left_right(dst_image, bboxes)

    #Randomly distort the colors. There are 4 ways to do it.
    dst_image = process.apply_with_random_selector(dst_image,
                                                   lambda x, ordering: process.distort_color(x, ordering, fast_mode=False),
                                                   num_cases=4)

    ## constrain the bboxes to [0,1] ##
    bboxes = tf.maximum(bboxes,0.)
    bboxes = tf.minimum(bboxes,1.)
    return dst_image, labels, bboxes