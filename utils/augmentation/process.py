import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import tf_extended as tfe
import random

BBOX_CROP_OVERLAP = 0.3     # Minimum overlap to keep a bbox after cropping.

def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.

    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        lower = random.uniform(0.5,1.)
        bright = random.uniform(0.,200.)
        hue = random.uniform(0.,0.5)
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=bright / 255.)
                image = tf.image.random_saturation(image, lower=lower, upper=1.)
            else:
                image = tf.image.random_saturation(image, lower=lower, upper=1.)
                image = tf.image.random_brightness(image, max_delta=bright / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=bright / 255.)
                image = tf.image.random_saturation(image, lower=lower, upper=1.)
                #image = tf.image.random_hue(image, max_delta=hue)
                image = tf.image.random_contrast(image, lower=lower, upper=1.)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=lower, upper=1.)
                image = tf.image.random_brightness(image, max_delta=bright / 255.)
                image = tf.image.random_contrast(image, lower=lower, upper=1.)
                #image = tf.image.random_hue(image, max_delta=hue)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=lower, upper=1.)
                #image = tf.image.random_hue(image, max_delta=hue)
                image = tf.image.random_brightness(image, max_delta=bright / 255.)
                image = tf.image.random_saturation(image, lower=lower, upper=1.)
            elif color_ordering == 3:
                #image = tf.image.random_hue(image, max_delta=hue)
                image = tf.image.random_saturation(image, lower=lower, upper=1.)
                image = tf.image.random_contrast(image, lower=lower, upper=1.)
                image = tf.image.random_brightness(image, max_delta=bright / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
        return image


def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.5,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.2, 1.0),
                                max_attempts=200,
                                clip_bboxes=True,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=tf.expand_dims(bboxes, 0),
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)
        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])

        # Update bounding boxes: resize and filter out.
        bboxes = tfe.bboxes_resize(distort_bbox, bboxes)
        labels, bboxes = tfe.bboxes_filter_overlap(labels, bboxes,
                                                   threshold=BBOX_CROP_OVERLAP,
                                                   assign_negative=False)
        return cropped_image, labels, bboxes, distort_bbox