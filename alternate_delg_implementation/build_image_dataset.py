## Functions (with minor changes) from https://github.com/tensorflow/models/blob/master/research/delf/delf/python/training/build_image_dataset.py

from absl import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
from six import BytesIO

import tensorflow as tf

import tensorflow_hub as hub
from six.moves.urllib.request import urlopen

import os
import cv2
import skimage.io
import imgaug as ia
from imgaug import augmenters as iaa
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
import seaborn as sns
import csv

num_shards = 128
val_split_size = 0.2
seed = 0
output_dir = "/content/tmp/data"

def _get_all_image_files_and_labels(name, df, image_dir):
  """Process input and get the image file paths, image ids and the labels.

  Args:
    name: 'train' or 'test'.
    df: dataframe with id, uuid
    image_dir: directory that stores downloaded images.
  Returns:
    image_paths: the paths to all images in the image_dir.
    file_ids: the unique ids of images.
    labels: the landmark id of all images. When name='test', the returned labels
      will be an empty list.
  Raises:
    ValueError: if input name is not supported.
  """
  image_paths = tf.io.gfile.glob(os.path.join(image_dir, '*.jpg'))
  file_ids = [os.path.basename(os.path.normpath(f))[:-4] for f in image_paths]

  df_ids = set(df["id"].unique())

  for fid in file_ids:
    if fid + ".jpg" not in df_ids:
      pathname = os.path.abspath(os.path.join(image_dir, fid + ".jpg"))
      os.remove(pathname)
      
  image_paths = tf.io.gfile.glob(os.path.join(image_dir, '*.jpg'))
  file_ids = [os.path.basename(os.path.normpath(f))[:-4] for f in image_paths]

  if name == "train":
    df = df.set_index('id')
    labels = [int(df.loc[fid + ".jpg"]['uuid']) for fid in file_ids]
  elif name == "test":
    labels = []
  else:
    raise ValueError('Unsupported dataset split name: %s' % name)
    
  return image_paths, file_ids, labels

def relabel_classes(labels):
  # Relabel image labels to contiguous values.
  unique_labels = sorted(set(labels))
  relabeling = {label: index for index, label in enumerate(unique_labels)}
  new_labels = [relabeling[label] for label in labels]
  return new_labels, relabeling

def _process_image(filename):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.jpg'.

  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  Raises:
    ValueError: if parsed image has wrong number of dimensions or channels.
  """
  # Read the image file.
  with tf.io.gfile.GFile(filename, 'rb') as f:
    image_data = f.read()

  # Decode the RGB JPEG.
  image = tf.io.decode_jpeg(image_data, channels=3)

  # Check that image converted to RGB
  if len(image.shape) != 3:
    raise ValueError('The parsed image number of dimensions is not 3 but %d' %
                     (image.shape))
  height = image.shape[0]
  width = image.shape[1]
  if image.shape[2] != 3:
    raise ValueError('The parsed image channels is not 3 but %d' %
                     (image.shape[2]))

  return image_data, height, width

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_id, image_buffer, height, width, label=None):
  """Build an Example proto for the given inputs.

  Args:
    file_id: string, unique id of an image file, e.g., '97c0a12e07ae8dd5'.
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
    label: integer, the landmark id and prediction label.

  Returns:
    Example proto.
  """
  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'
  features = {
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace.encode('utf-8')),
      'image/channels': _int64_feature(channels),
      'image/format': _bytes_feature(image_format.encode('utf-8')),
      'image/id': _bytes_feature(file_id.encode('utf-8')),
      'image/encoded': _bytes_feature(image_buffer)
  }
  if label is not None:
    features['image/class/label'] = _int64_feature(label)
  example = tf.train.Example(features=tf.train.Features(feature=features))

  return example

def _write_tfrecord(output_prefix, image_paths, file_ids, labels):
  """Read image files and write image and label data into TFRecord files.

  Args:
    output_prefix: string, the prefix of output files, e.g. 'train'.
    image_paths: list of strings, the paths to images to be converted.
    file_ids: list of strings, the image unique ids.
    labels: list of integers, the landmark ids of images. It is an empty list
      when output_prefix='test'.

  Raises:
    ValueError: if the length of input images, ids and labels don't match
  """
  if output_prefix == "test":
    labels = [None] * len(image_paths)
  if not len(image_paths) == len(file_ids) == len(labels):
    raise ValueError('length of image_paths, file_ids, labels shoud be the' +
                     ' same. But they are %d, %d, %d, respectively' %
                     (len(image_paths), len(file_ids), len(labels)))

  spacing = np.linspace(0, len(image_paths), num_shards + 1, dtype=np.int)

  for shard in range(num_shards):
    output_file = os.path.join(
        output_dir,
        '%s-%.5d-of-%.5d' % (output_prefix, shard, num_shards))
    writer = tf.io.TFRecordWriter(output_file)
    print('Processing shard ', shard, ' and writing file ', output_file)
    for i in range(spacing[shard], spacing[shard + 1]):
      image_buffer, height, width = _process_image(image_paths[i])
      example = _convert_to_example(file_ids[i], image_buffer, height, width,
                                    labels[i])
      writer.write(example.SerializeToString())
    writer.close()

def _write_relabeling_rules(relabeling_rules):
  """Write to a file the relabeling rules when the clean train dataset is used.

  Args:
    relabeling_rules: dictionary of relabeling rules applied when the clean
      train dataset is used (key = old_label, value = new_label).
  """
  relabeling_file_name = os.path.join(output_dir, 'relabeling.csv')
  with tf.io.gfile.GFile(relabeling_file_name, 'w') as relabeling_file:
    csv_writer = csv.writer(relabeling_file, delimiter=',')
    csv_writer.writerow(['new_label', 'old_label'])
    for old_label, new_label in relabeling_rules.items():
      csv_writer.writerow([new_label, old_label])


def _shuffle_by_columns(np_array, random_state):
  """Shuffle the columns of a 2D numpy array.

  Args:
    np_array: array to shuffle.
    random_state: numpy RandomState to be used for shuffling.
  Returns:
    The shuffled array.
  """
  columns = np_array.shape[1]
  columns_indices = np.arange(columns)
  random_state.shuffle(columns_indices)
  return np_array[:, columns_indices]

def _build_train_and_validation_splits(image_paths, file_ids, labels,
                                       validation_split_size, seed):
  """Create TRAIN and VALIDATION splits containg all labels in equal proportion.

  Args:
    image_paths: list of paths to the image files in the train dataset.
    file_ids: list of image file ids in the train dataset.
    labels: list of image labels in the train dataset.
    validation_split_size: size of the VALIDATION split as a ratio of the train
      dataset.
    seed: seed to use for shuffling the dataset for reproducibility purposes.

  Returns:
    splits : tuple containing the TRAIN and VALIDATION splits.
  Raises:
    ValueError: if the image attributes arrays don't all have the same length,
                which makes the shuffling impossible.
  """
  # Ensure all image attribute arrays have the same length.
  total_images = len(file_ids)
  if not (len(image_paths) == total_images and len(labels) == total_images):
    raise ValueError('Inconsistencies between number of file_ids (%d), number '
                     'of image_paths (%d) and number of labels (%d). Cannot'
                     'shuffle the train dataset.'% (total_images,
                                                    len(image_paths),
                                                    len(labels)))

  # Stack all image attributes arrays in a single 2D array of dimensions
  # (3, number of images) and group by label the indices of datapoins in the
  # image attributes arrays. Explicitly convert label types from 'int' to 'str'
  # to avoid implicit conversion during stacking with image_paths and file_ids
  # which are 'str'.
  labels_str = [str(label) for label in labels]
  image_attrs = np.stack((image_paths, file_ids, labels_str))
  image_attrs_idx_by_label = {}

  print(labels)

  for index, label in enumerate(labels):
    if label not in image_attrs_idx_by_label:
      image_attrs_idx_by_label[label] = []
    image_attrs_idx_by_label[label].append(index)

  # Create subsets of image attributes by label, shuffle them separately and
  # split each subset into TRAIN and VALIDATION splits based on the size of the
  # validation split.
  splits = {
      "validation": [],
      "train": []
  }
  rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))

  for label, indexes in image_attrs_idx_by_label.items():
    # Create the subset for the current label.
    image_attrs_label = image_attrs[:, indexes]
    # Shuffle the current label subset.
    image_attrs_label = _shuffle_by_columns(image_attrs_label, rs)
    # Split the current label subset into TRAIN and VALIDATION splits and add
    # each split to the list of all splits.
    images_per_label = image_attrs_label.shape[1]
    cutoff_idx = max(1, int(validation_split_size * images_per_label))
    splits["validation"].append(image_attrs_label[:, 0 : cutoff_idx])
    splits["train"].append(image_attrs_label[:, cutoff_idx : ])

  print(image_attrs_idx_by_label)

  # Concatenate all subsets of image attributes into TRAIN and VALIDATION splits
  # and reshuffle them again to ensure variance of labels across batches.
  validation_split = _shuffle_by_columns(
      np.concatenate(splits["validation"], axis=1), rs)
  train_split = _shuffle_by_columns(
      np.concatenate(splits["train"], axis=1), rs)

  # Unstack the image attribute arrays in the TRAIN and VALIDATION splits and
  # convert them back to lists. Convert labels back to 'int' from 'str'
  # following the explicit type change from 'str' to 'int' for stacking.
  return (
      {
          "image_paths": validation_split[0, :].tolist(),
          "file_ids": validation_split[1, :].tolist(),
          "labels": [int(label) for label in validation_split[2, :].tolist()]
      }, {
          "image_paths": train_split[0, :].tolist(),
          "file_ids": train_split[1, :].tolist(),
          "labels": [int(label) for label in train_split[2, :].tolist()]
      })

def _build_train_tfrecord_dataset(df,
                                  image_dir,
                                  generate_train_validation_splits,
                                  validation_split_size,
                                  seed):
  """Build a TFRecord dataset for the train split.

  Args:
    df: dataframe with id, uuid.
    image_dir: directory that stores downloaded images.
    generate_train_validation_splits: whether to split the test dataset into
      TRAIN and VALIDATION splits.
    validation_split_size: size of the VALIDATION split as a ratio of the train
      dataset. Only used if 'generate_train_validation_splits' is True.
    seed: seed to use for shuffling the dataset for reproducibility purposes.
      Only used if 'generate_train_validation_splits' is True.

  Returns:
    Nothing. After the function call, sharded TFRecord files are materialized.
  Raises:
    ValueError: if the size of the VALIDATION split is outside (0,1) when TRAIN
                and VALIDATION splits need to be generated.
  """
  # Make sure the size of the VALIDATION split is inside (0, 1) if we need to
  # generate the TRAIN and VALIDATION splits.
  if generate_train_validation_splits:
    if validation_split_size <= 0 or validation_split_size >= 1:
      raise ValueError('Invalid VALIDATION split size. Expected inside (0,1)'
                        'but received %f.' % validation_split_size)

  # Load all train images.
  image_paths, file_ids, labels = _get_all_image_files_and_labels(
      "train", df, image_dir)
  
  new_labels, relabeling_rules = relabel_classes(labels)
  _write_relabeling_rules(relabeling_rules)

  if generate_train_validation_splits:
    # Generate the TRAIN and VALIDATION splits and write them to TFRecord.
    validation_split, train_split = _build_train_and_validation_splits(
        image_paths, file_ids, labels, validation_split_size, seed)
    _write_tfrecord("validation",
                    validation_split["image_paths"],
                    validation_split["file_ids"],
                    validation_split["labels"])
    _write_tfrecord("train",
                    train_split["image_paths"],
                    train_split["file_ids"],
                    train_split["labels"])
  else:
    # Write to TFRecord a single split, TRAIN.
    _write_tfrecord("train", image_paths, file_ids, labels)

def _build_test_tfrecord_dataset(df, image_dir):
  """Build a TFRecord dataset for the 'test' split.

  Args:
    df: dataframe with id, uuid.
    image_dir: directory that stores downloaded images.

  Returns:
    Nothing. After the function call, sharded TFRecord files are materialized.
  """
  image_paths, file_ids, labels = _get_all_image_files_and_labels(
    "test", df, image_dir)
  _write_tfrecord("test", image_paths, file_ids, labels)
