## In this file we attempt to extract the data from the tensorflow file in Menon et al.
import tensorflow as tf

import numpy as np
import IPython.display as display
import os

filenames = ['/Users/thomasblake/Coding/diss/logit_adjustment/data/cifar10_test.tfrecord']
raw_dataset = tf.data.TFRecordDataset(filenames)
for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)



def _process_image(record, training):
  """Decodes the image and performs data augmentation if training."""
  image = tf.io.decode_raw(record, tf.uint8)
  image = tf.cast(image, tf.float32)
  image = tf.reshape(image, [32, 32, 3])
  image = image * (1. / 255) - 0.5
  if training:
    padding = 4
    image = tf.image.resize_with_crop_or_pad(image, 32 + padding, 32 + padding)
    image = tf.image.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)
  return image


def _parse(serialized_examples, training):
  """Parses the given protos and performs data augmentation if training."""
  feature_spec = {
      'image/encoded': tf.io.FixedLenFeature((), tf.string),
      'image/class/label': tf.io.FixedLenFeature((), tf.int64)
  }
  features = tf.io.parse_example(serialized_examples, feature_spec)
  images = tf.map_fn(
      lambda record: _process_image(record, training),
      features['image/encoded'],
      dtype=tf.float32)
  return images, features['image/class/label']


def create_tf_dataset(path, batch_size, training):
  """Creates a Tensorflow Dataset instance for training/testing.

  Args:
    dataset:    Dataset definition.
    data_home:  Directory where the .tfrecord files are stored.
    batch_size: Batch size.
    training:   Whether to return a training dataset or not. Training datasets
      have data augmentation.

  Returns:
    A tf.data.Dataset instance.
  """


  return tf.data.TFRecordDataset(
      path
  ).shuffle(
      10000
  ).batch(
      batch_size, drop_remainder=training
  ).map(
      lambda record: _parse(record, training)
  ).prefetch(
      tf.data.experimental.AUTOTUNE
  )

testDataset = create_tf_dataset('logit_adjustment/data/cifar10_test.tfrecord', 1000, False)

# for batch in testDataset:
#   print(batch)

print(len(testDataset))