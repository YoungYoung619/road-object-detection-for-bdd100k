# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Convert a dataset to TFRecords format, which can be easily integrated into
a TensorFlow pipeline.

Usage:
```shell
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=/tmp/pascalvoc \
    --output_name=pascalvoc \
    --output_dir=/tmp/
```
"""
import tensorflow as tf

from dataset import pascalvoc_to_tfrecords

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name', 'bdd100k',
    'The name of the dataset to convert.')
tf.app.flags.DEFINE_string(
    'dataset_dir', "h:/Data/BDD100K/bdd/images/100k/",
    'Directory where the original dataset is stored.')
tf.app.flags.DEFINE_string(
    'output_name', 'bdd100k_val',
    'Basename used for TFRecords output files.')
tf.app.flags.DEFINE_string(
    'output_dir', './dataset/bdd100k_TfRecord',
    'Output directory where to store TFRecords files.')


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    print('Dataset directory:', FLAGS.dataset_dir)
    print('Output directory:', FLAGS.output_dir)

    if FLAGS.dataset_name == 'pascalvoc'or FLAGS.dataset_name == "bdd100k":
        pascalvoc_to_tfrecords.run(FLAGS.dataset_dir, FLAGS.output_dir, FLAGS.output_name)
    else:
        raise ValueError('Dataset [%s] was not recognized.' % FLAGS.dataset_name)

if __name__ == '__main__':
    tf.app.run()

