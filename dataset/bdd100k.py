#-*-coding:utf-8-*-
"""Provides data for the Pascal VOC Dataset (images + annotations).
"""
import tensorflow as tf
from dataset import pascalvoc_common

slim = tf.contrib.slim

FILE_PATTERN = 'bdd100k_%s_*.tfrecord'
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
}

SPLITS_TO_SIZES = {
    'train': 70000,
}

NUM_CLASSES = 10

BDD100K_LABELS = {
    'none': (0, 'Background'),
    'bus': (1, 'Vehicle'),
    'traffic light': (2, 'Flag'),
    'traffic sign': (3, 'Flag'),
    'person': (4, 'Human'),
    'bike': (5, 'Vehicle'),
    'truck': (6, 'Vehicle'),
    'motor': (7, 'Vehicle'),
    'car': (8, 'Vehicle'),
    'train': (9, 'Vehicle'),
    'rider': (10, 'Human'),
}

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading ImageNet.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if not file_pattern:
        file_pattern = FILE_PATTERN
    return pascalvoc_common.get_split(split_name, dataset_dir,
                                      file_pattern, reader,
                                      SPLITS_TO_SIZES,
                                      ITEMS_TO_DESCRIPTIONS,
                                      NUM_CLASSES)