import tensorflow as tf

import functools
import os
import tempfile

from absl import logging
import h5py
import math

## Defining gem pooling layer
def gem(x, axis=None, power=3., eps=1e-6):
  """Performs generalized mean pooling (GeM).

  Args:
    x: [B, H, W, D] A float32 Tensor.
    axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
    power: Float, power > 0 is an inverse exponent parameter (GeM power).
    eps: Float, parameter for numerical stability.

  Returns:
    output: [B, D] A float32 Tensor.
  """
  if axis is None:
    axis = [1, 2]
  tmp = tf.pow(tf.maximum(x, eps), power)
  out = tf.pow(tf.reduce_mean(tmp, axis=axis, keepdims=False), 1. / power)
  return out

## Defining Resnet-50
layers = tf.keras.layers

class _IdentityBlock(tf.keras.Model):
  """_IdentityBlock is the block that has no conv layer at shortcut.

  Args:
    kernel_size: the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    data_format: data_format for the input ('channels_first' or
      'channels_last').
  """

  def __init__(self, kernel_size, filters, stage, block, data_format):
    super(_IdentityBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_axis = 1 if data_format == 'channels_first' else 3

    self.conv2a = layers.Conv2D(
        filters1, (1, 1), name=conv_name_base + '2a', data_format=data_format)
    self.bn2a = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2a')

    self.conv2b = layers.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        data_format=data_format,
        name=conv_name_base + '2b')
    self.bn2b = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2b')

    self.conv2c = layers.Conv2D(
        filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format)
    self.bn2c = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2c')

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)

class _ConvBlock(tf.keras.Model):
  """_ConvBlock is the block that has a conv layer at shortcut.

  Args:
      kernel_size: the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      data_format: data_format for the input ('channels_first' or
        'channels_last').
      strides: strides for the convolution. Note that from stage 3, the first
        conv layer at main path is with strides=(2,2), and the shortcut should
        have strides=(2,2) as well.
  """

  def __init__(self,
               kernel_size,
               filters,
               stage,
               block,
               data_format,
               strides=(2, 2)):
    super(_ConvBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_axis = 1 if data_format == 'channels_first' else 3

    self.conv2a = layers.Conv2D(
        filters1, (1, 1),
        strides=strides,
        name=conv_name_base + '2a',
        data_format=data_format)
    self.bn2a = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2a')

    self.conv2b = layers.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        name=conv_name_base + '2b',
        data_format=data_format)
    self.bn2b = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2b')

    self.conv2c = layers.Conv2D(
        filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format)
    self.bn2c = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2c')

    self.conv_shortcut = layers.Conv2D(
        filters3, (1, 1),
        strides=strides,
        name=conv_name_base + '1',
        data_format=data_format)
    self.bn_shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    shortcut = self.conv_shortcut(input_tensor)
    shortcut = self.bn_shortcut(shortcut, training=training)

    x += shortcut
    return tf.nn.relu(x)

# pylint: disable=not-callable
class ResNet50(tf.keras.Model):
  """Instantiates the ResNet50 architecture.

  Args:
    data_format: format for the image. Either 'channels_first' or
      'channels_last'.  'channels_first' is typically faster on GPUs while
      'channels_last' is typically faster on CPUs. See
      https://www.tensorflow.org/performance/performance_guide#data_formats
    name: Prefix applied to names of variables created in the model.
    include_top: whether to include the fully-connected layer at the top of the
      network.
    pooling: Optional pooling mode for feature extraction when `include_top` is
      False. 'None' means that the output of the model will be the 4D tensor
      output of the last convolutional layer. 'avg' means that global average
      pooling will be applied to the output of the last convolutional layer, and
      thus the output of the model will be a 2D tensor. 'max' means that global
      max pooling will be applied. 'gem' means GeM pooling will be applied.
    block3_strides: whether to add a stride of 2 to block3 to make it compatible
      with tf.slim ResNet implementation.
    average_pooling: whether to do average pooling of block4 features before
      global pooling.
    classes: optional number of classes to classify images into, only to be
      specified if `include_top` is True.
    gem_power: GeM power for GeM pooling. Only used if pooling == 'gem'.
    embedding_layer: whether to create an embedding layer (FC whitening layer).
    embedding_layer_dim: size of the embedding layer.

  Raises:
      ValueError: in case of invalid argument for data_format.
  """

  def __init__(self,
               data_format,
               name='',
               include_top=True,
               pooling=None,
               block3_strides=False,
               average_pooling=True,
               classes=1000,
               gem_power=3.0,
               embedding_layer=False,
               embedding_layer_dim=2048):
    super(ResNet50, self).__init__(name=name)

    valid_channel_values = ('channels_first', 'channels_last')
    if data_format not in valid_channel_values:
      raise ValueError('Unknown data_format: %s. Valid values: %s' %
                       (data_format, valid_channel_values))
    self.include_top = include_top
    self.block3_strides = block3_strides
    self.average_pooling = average_pooling
    self.pooling = pooling

    def conv_block(filters, stage, block, strides=(2, 2)):
      return _ConvBlock(
          3,
          filters,
          stage=stage,
          block=block,
          data_format=data_format,
          strides=strides)

    def id_block(filters, stage, block):
      return _IdentityBlock(
          3, filters, stage=stage, block=block, data_format=data_format)

    self.conv1 = layers.Conv2D(
        64, (7, 7),
        strides=(2, 2),
        data_format=data_format,
        padding='same',
        name='conv1')
    bn_axis = 1 if data_format == 'channels_first' else 3
    self.bn_conv1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')
    self.max_pool = layers.MaxPooling2D((3, 3),
                                        strides=(2, 2),
                                        data_format=data_format)

    self.l2a = conv_block([64, 64, 256], stage=2, block='a', strides=(1, 1))
    self.l2b = id_block([64, 64, 256], stage=2, block='b')
    self.l2c = id_block([64, 64, 256], stage=2, block='c')

    self.l3a = conv_block([128, 128, 512], stage=3, block='a')
    self.l3b = id_block([128, 128, 512], stage=3, block='b')
    self.l3c = id_block([128, 128, 512], stage=3, block='c')
    self.l3d = id_block([128, 128, 512], stage=3, block='d')

    self.l4a = conv_block([256, 256, 1024], stage=4, block='a')
    self.l4b = id_block([256, 256, 1024], stage=4, block='b')
    self.l4c = id_block([256, 256, 1024], stage=4, block='c')
    self.l4d = id_block([256, 256, 1024], stage=4, block='d')
    self.l4e = id_block([256, 256, 1024], stage=4, block='e')
    self.l4f = id_block([256, 256, 1024], stage=4, block='f')

    # Striding layer that can be used on top of block3 to produce feature maps
    # with the same resolution as the TF-Slim implementation.
    if self.block3_strides:
      self.subsampling_layer = layers.MaxPooling2D((1, 1),
                                                   strides=(2, 2),
                                                   data_format=data_format)
      self.l5a = conv_block([512, 512, 2048],
                            stage=5,
                            block='a',
                            strides=(1, 1))
    else:
      self.l5a = conv_block([512, 512, 2048], stage=5, block='a')
    self.l5b = id_block([512, 512, 2048], stage=5, block='b')
    self.l5c = id_block([512, 512, 2048], stage=5, block='c')

    self.avg_pool = layers.AveragePooling2D((7, 7),
                                            strides=(7, 7),
                                            data_format=data_format)

    if self.include_top:
      self.flatten = layers.Flatten()
      self.fc1000 = layers.Dense(classes, name='fc1000')
    else:
      reduction_indices = [1, 2] if data_format == 'channels_last' else [2, 3]
      reduction_indices = tf.constant(reduction_indices)
      if pooling == 'avg':
        self.global_pooling = functools.partial(
            tf.reduce_mean, axis=reduction_indices, keepdims=False)
      elif pooling == 'max':
        self.global_pooling = functools.partial(
            tf.reduce_max, axis=reduction_indices, keepdims=False)
      elif pooling == 'gem':
        logging.info('Adding GeMPooling layer with power %f', gem_power)
        self.global_pooling = functools.partial(
            gem, axis=reduction_indices, power=gem_power)
      else:
        self.global_pooling = None
      if embedding_layer:
        logging.info('Adding embedding layer with dimension %d',
                     embedding_layer_dim)
        self.embedding_layer = layers.Dense(
            embedding_layer_dim, name='embedding_layer')
      else:
        self.embedding_layer = None

  def build_call(self, inputs, training=True, intermediates_dict=None):
    """Building the ResNet50 model.

    Args:
      inputs: Images to compute features for.
      training: Whether model is in training phase.
      intermediates_dict: `None` or dictionary. If not None, accumulate feature
        maps from intermediate blocks into the dictionary. ""

    Returns:
      Tensor with featuremap.
    """

    x = self.conv1(inputs)
    x = self.bn_conv1(x, training=training)
    x = tf.nn.relu(x)
    if intermediates_dict is not None:
      intermediates_dict['block0'] = x

    x = self.max_pool(x)
    if intermediates_dict is not None:
      intermediates_dict['block0mp'] = x

    # Block 1 (equivalent to "conv2" in Resnet paper).
    x = self.l2a(x, training=training)
    x = self.l2b(x, training=training)
    x = self.l2c(x, training=training)
    if intermediates_dict is not None:
      intermediates_dict['block1'] = x

    # Block 2 (equivalent to "conv3" in Resnet paper).
    x = self.l3a(x, training=training)
    x = self.l3b(x, training=training)
    x = self.l3c(x, training=training)
    x = self.l3d(x, training=training)
    if intermediates_dict is not None:
      intermediates_dict['block2'] = x

    # Block 3 (equivalent to "conv4" in Resnet paper).
    x = self.l4a(x, training=training)
    x = self.l4b(x, training=training)
    x = self.l4c(x, training=training)
    x = self.l4d(x, training=training)
    x = self.l4e(x, training=training)
    x = self.l4f(x, training=training)

    if self.block3_strides:
      x = self.subsampling_layer(x)
      if intermediates_dict is not None:
        intermediates_dict['block3'] = x
    else:
      if intermediates_dict is not None:
        intermediates_dict['block3'] = x

    x = self.l5a(x, training=training)
    x = self.l5b(x, training=training)
    x = self.l5c(x, training=training)

    if self.average_pooling:
      x = self.avg_pool(x)
      if intermediates_dict is not None:
        intermediates_dict['block4'] = x
    else:
      if intermediates_dict is not None:
        intermediates_dict['block4'] = x

    if self.include_top:
      return self.fc1000(self.flatten(x))
    elif self.global_pooling:
      x = self.global_pooling(x)
      if self.embedding_layer:
        x = self.embedding_layer(x)
      return x
    else:
      return x

  def call(self, inputs, training=True, intermediates_dict=None):
    """Call the ResNet50 model.

    Args:
      inputs: Images to compute features for.
      training: Whether model is in training phase.
      intermediates_dict: `None` or dictionary. If not None, accumulate feature
        maps from intermediate blocks into the dictionary. ""

    Returns:
      Tensor with featuremap.
    """
    return self.build_call(inputs, training, intermediates_dict)

  def restore_weights(self, filepath):
    """Load pretrained weights.

    This function loads a .h5 file from the filepath with saved model weights
    and assigns them to the model.

    Args:
      filepath: String, path to the .h5 file

    Raises:
      ValueError: if the file referenced by `filepath` does not exist.
    """
    if not tf.io.gfile.exists(filepath):
      raise ValueError('Unable to load weights from %s. You must provide a'
                       'valid file.' % (filepath))

    # Create a local copy of the weights file for h5py to be able to read it.
    local_filename = os.path.basename(filepath)
    tmp_filename = os.path.join(tempfile.gettempdir(), local_filename)
    tf.io.gfile.copy(filepath, tmp_filename, overwrite=True)

    # Load the content of the weights file.
    f = h5py.File(tmp_filename, mode='r')
    saved_layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

    try:
      # Iterate through all the layers assuming the max `depth` is 2.
      for layer in self.layers:
        if hasattr(layer, 'layers'):
          for inlayer in layer.layers:
            # Make sure the weights are in the saved model, and that we are in
            # the innermost layer.
            if inlayer.name not in saved_layer_names:
              raise ValueError('Layer %s absent from the pretrained weights.'
                               'Unable to load its weights.' % (inlayer.name))
            if hasattr(inlayer, 'layers'):
              raise ValueError('Layer %s is not a depth 2 layer. Unable to load'
                               'its weights.' % (inlayer.name))
            # Assign the weights in the current layer.
            g = f[inlayer.name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            weight_values = [g[weight_name] for weight_name in weight_names]
            logging.info('Setting the weights for layer %s', inlayer.name)
            inlayer.set_weights(weight_values)
    finally:
      # Clean up the temporary file.
      tf.io.gfile.remove(tmp_filename)

  def log_weights(self):
    """Log backbone weights."""
    logging.info('Logging backbone weights')
    logging.info('------------------------')
    for layer in self.layers:
      if hasattr(layer, 'layers'):
        for inlayer in layer.layers:
          logging.info('Weights for layer: %s, inlayer % s', layer.name,
                       inlayer.name)
          weights = inlayer.get_weights()
          logging.info(weights)
      else:
        logging.info('Layer %s does not have inner layers.', layer.name)

## DELF Model Defintion

reg = tf.keras.regularizers
_DECAY = 0.0001

class AttentionModel(tf.keras.Model):
  """Instantiates attention model.

  Uses two [kernel_size x kernel_size] convolutions and softplus as activation
  to compute an attention map with the same resolution as the featuremap.
  Features l2-normalized and aggregated using attention probabilites as weights.
  The features (targets) to be aggregated can be the input featuremap, or a
  different one with the same resolution.
  """

  def __init__(self, kernel_size=1, decay=_DECAY, name='attention'):
    """Initialization of attention model.

    Args:
      kernel_size: int, kernel size of convolutions.
      decay: float, decay for l2 regularization of kernel weights.
      name: str, name to identify model.
    """
    super(AttentionModel, self).__init__(name=name)

    # First convolutional layer (called with relu activation).
    self.conv1 = layers.Conv2D(
        512,
        kernel_size,
        kernel_regularizer=reg.l2(decay),
        padding='same',
        name='attn_conv1')
    self.bn_conv1 = layers.BatchNormalization(axis=3, name='bn_conv1')

    # Second convolutional layer, with softplus activation.
    self.conv2 = layers.Conv2D(
        1,
        kernel_size,
        kernel_regularizer=reg.l2(decay),
        padding='same',
        name='attn_conv2')
    self.activation_layer = layers.Activation('softplus')

  def call(self, inputs, targets=None, training=True):
    x = self.conv1(inputs)
    x = self.bn_conv1(x, training=training)
    x = tf.nn.relu(x)

    score = self.conv2(x)
    prob = self.activation_layer(score)

    # Aggregate inputs if targets is None.
    if targets is None:
      targets = inputs

    # L2-normalize the featuremap before pooling.
    targets = tf.nn.l2_normalize(targets, axis=-1)
    feat = tf.reduce_mean(tf.multiply(targets, prob), [1, 2], keepdims=False)

    return feat, prob, score

class AutoencoderModel(tf.keras.Model):
  """Instantiates the Keras Autoencoder model."""

  def __init__(self, reduced_dimension, expand_dimension, kernel_size=1,
               name='autoencoder'):
    """Initialization of Autoencoder model.

    Args:
      reduced_dimension: int, the output dimension of the autoencoder layer.
      expand_dimension: int, the input dimension of the autoencoder layer.
      kernel_size: int or tuple, height and width of the 2D convolution window.
      name: str, name to identify model.
    """
    super(AutoencoderModel, self).__init__(name=name)
    self.conv1 = layers.Conv2D(
        reduced_dimension,
        kernel_size,
        padding='same',
        name='autoenc_conv1')
    self.conv2 = layers.Conv2D(
        expand_dimension,
        kernel_size,
        activation=tf.keras.activations.relu,
        padding='same',
        name='autoenc_conv2')

  def call(self, inputs):
    dim_reduced_features = self.conv1(inputs)
    dim_expanded_features = self.conv2(dim_reduced_features)
    return dim_expanded_features, dim_reduced_features

class Delf(tf.keras.Model):
  """Instantiates Keras DELF model using ResNet50 as backbone.

  This class implements the [DELF](https://arxiv.org/abs/1612.06321) model for
  extracting local features from images. The backbone is a ResNet50 network
  that extracts featuremaps from both conv_4 and conv_5 layers. Activations
  from conv_4 are used to compute an attention map of the same resolution.
  """

  def __init__(self,
               block3_strides=True,
               name='DELF',
               pooling='avg',
               gem_power=3.0,
               embedding_layer=False,
               embedding_layer_dim=2048,
               use_dim_reduction=False,
               reduced_dimension=128,
               dim_expand_channels=1024):
    """Initialization of DELF model.

    Args:
      block3_strides: bool, whether to add strides to the output of block3.
      name: str, name to identify model.
      pooling: str, pooling mode for global feature extraction; possible values
        are 'None', 'avg', 'max', 'gem.'
      gem_power: float, GeM power for GeM pooling. Only used if pooling ==
        'gem'.
      embedding_layer: bool, whether to create an embedding layer (FC whitening
        layer).
      embedding_layer_dim: int, size of the embedding layer.
      use_dim_reduction: Whether to integrate dimensionality reduction layers.
        If True, extra layers are added to reduce the dimensionality of the
        extracted features.
      reduced_dimension: int, only used if use_dim_reduction is True. The output
        dimension of the autoencoder layer.
      dim_expand_channels: int, only used if use_dim_reduction is True. The
        number of channels of the backbone block used. Default value 1024 is the
        number of channels of backbone block 'block3'.
    """
    super(Delf, self).__init__(name=name)

    # Backbone using Keras ResNet50.
    self.backbone = ResNet50(
        'channels_last',
        name='backbone',
        include_top=False,
        pooling=pooling,
        block3_strides=block3_strides,
        average_pooling=False,
        gem_power=gem_power,
        embedding_layer=embedding_layer,
        embedding_layer_dim=embedding_layer_dim)

    # Attention model.
    self.attention = AttentionModel(name='attention')

    # Autoencoder model.
    self._use_dim_reduction = use_dim_reduction
    if self._use_dim_reduction:
      self.autoencoder = AutoencoderModel(reduced_dimension,
                                          dim_expand_channels,
                                          name='autoencoder')

  def init_classifiers(self, num_classes, desc_classification=None):
    """Define classifiers for training backbone and attention models."""
    self.num_classes = num_classes
    if desc_classification is None:
      self.desc_classification = layers.Dense(
          num_classes, activation=None, kernel_regularizer=None, name='desc_fc')
    else:
      self.desc_classification = desc_classification
    self.attn_classification = layers.Dense(
        num_classes, activation=None, kernel_regularizer=None, name='att_fc')

  def global_and_local_forward_pass(self, images, training=True):
    """Run a forward to calculate global descriptor and attention prelogits.

    Args:
      images: Tensor containing the dataset on which to run the forward pass.
      training: Indicator of wether the forward pass is running in training mode
        or not.

    Returns:
      Global descriptor prelogits, attention prelogits, attention scores,
        backbone weights.
    """

    backbone_blocks = {}
    desc_prelogits = self.backbone.build_call(
        images, intermediates_dict=backbone_blocks, training=training)
    # Prevent gradients from propagating into the backbone. See DELG paper:
    # https://arxiv.org/abs/2001.05027.
    block3 = backbone_blocks['block3']  # pytype: disable=key-error
    block3 = tf.stop_gradient(block3)
    if self._use_dim_reduction:
      (dim_expanded_features, dim_reduced_features) = self.autoencoder(block3)
      attn_prelogits, attn_scores, _ = self.attention(
          block3,
          targets=dim_expanded_features,
          training=training)
    else:
      attn_prelogits, attn_scores, _ = self.attention(block3, training=training)
      dim_expanded_features = None
      dim_reduced_features = None
    return (desc_prelogits, attn_prelogits, attn_scores, backbone_blocks,
            dim_expanded_features, dim_reduced_features)

  def build_call(self, input_image, training=True):
    (global_feature, _, attn_scores, backbone_blocks, _,
     dim_reduced_features) = self.global_and_local_forward_pass(input_image,
                                                                training)
    if self._use_dim_reduction:
      features = dim_reduced_features
    else:
      features = backbone_blocks['block3']  # pytype: disable=key-error
    return global_feature, attn_scores, features

  def call(self, input_image, training=True):
    _, probs, features = self.build_call(input_image, training=training)
    return probs, features


def cosine_classifier_logits(prelogits,
                             labels,
                             num_classes,
                             cosine_weights,
                             scale_factor,
                             arcface_margin,
                             training=True):
  """Compute cosine classifier logits using ArFace margin.

  Args:
    prelogits: float tensor of shape [batch_size, embedding_layer_dim].
    labels: int tensor of shape [batch_size].
    num_classes: int, number of classes.
    cosine_weights: float tensor of shape [embedding_layer_dim, num_classes].
    scale_factor: float.
    arcface_margin: float. Only used if greater than zero, and training is True.
    training: bool, True if training, False if eval.

  Returns:
    logits: Float tensor [batch_size, num_classes].
  """
  # L2-normalize prelogits, then obtain cosine similarity.
  normalized_prelogits = tf.math.l2_normalize(prelogits, axis=1)
  normalized_weights = tf.math.l2_normalize(cosine_weights, axis=0)
  cosine_sim = tf.matmul(normalized_prelogits, normalized_weights)

  # Optionally use ArcFace margin.
  if training and arcface_margin > 0.0:
    # Reshape labels tensor from [batch_size] to [batch_size, num_classes].
    one_hot_labels = tf.one_hot(labels, num_classes)
    cosine_sim = apply_arcface_margin(cosine_sim,
                                      one_hot_labels,
                                      arcface_margin)

  # Apply the scale factor to logits and return.
  logits = scale_factor * cosine_sim
  return logits


def apply_arcface_margin(cosine_sim, one_hot_labels, arcface_margin):
  """Applies ArcFace margin to cosine similarity inputs.

  For a reference, see https://arxiv.org/pdf/1801.07698.pdf. ArFace margin is
  applied to angles from correct classes (as per the ArcFace paper), and only
  if they are <= (pi - margin). Otherwise, applying the margin may actually
  improve their cosine similarity.

  Args:
    cosine_sim: float tensor with shape [batch_size, num_classes].
    one_hot_labels: int tensor with shape [batch_size, num_classes].
    arcface_margin: float.

  Returns:
    cosine_sim_with_margin: Float tensor with shape [batch_size, num_classes].
  """
  theta = tf.acos(cosine_sim, name='acos')
  selected_labels = tf.where(tf.greater(theta, math.pi - arcface_margin),
                             tf.zeros_like(one_hot_labels),
                             one_hot_labels,
                             name='selected_labels')
  final_theta = tf.where(tf.cast(selected_labels, dtype=tf.bool),
                         theta + arcface_margin,
                         theta,
                         name='final_theta')
  return tf.cos(final_theta, name='cosine_sim_with_margin')


class Delg(Delf):
  """Instantiates Keras DELG model using ResNet50 as backbone.

  This class implements the [DELG](https://arxiv.org/abs/2001.05027) model for
  extracting local and global features from images. The same attention layer
  is trained as in the DELF model. In addition, the extraction of global
  features is trained using GeMPooling, a FC whitening layer also called
  "embedding layer" and ArcFace loss.
  """

  def __init__(self,
               block3_strides=True,
               name='DELG',
               gem_power=3.0,
               embedding_layer_dim=2048,
               scale_factor_init=45.25,  # sqrt(2048)
               arcface_margin=0.1,
               use_dim_reduction=False,
               reduced_dimension=128,
               dim_expand_channels=1024):
    """Initialization of DELG model.

    Args:
      block3_strides: bool, whether to add strides to the output of block3.
      name: str, name to identify model.
      gem_power: float, GeM power parameter.
      embedding_layer_dim : int, dimension of the embedding layer.
      scale_factor_init: float.
      arcface_margin: float, ArcFace margin.
      use_dim_reduction: Whether to integrate dimensionality reduction layers.
        If True, extra layers are added to reduce the dimensionality of the
        extracted features.
      reduced_dimension: Only used if use_dim_reduction is True, the output
        dimension of the dim_reduction layer.
      dim_expand_channels: Only used if use_dim_reduction is True, the
        number of channels of the backbone block used. Default value 1024 is the
        number of channels of backbone block 'block3'.
    """
    logging.info('Creating Delg model, gem_power %d, embedding_layer_dim %d',
                 gem_power, embedding_layer_dim)
    super(Delg, self).__init__(block3_strides=block3_strides,
                               name=name,
                               pooling='gem',
                               gem_power=gem_power,
                               embedding_layer=True,
                               embedding_layer_dim=embedding_layer_dim,
                               use_dim_reduction=use_dim_reduction,
                               reduced_dimension=reduced_dimension,
                               dim_expand_channels=dim_expand_channels)
    self._embedding_layer_dim = embedding_layer_dim
    self._scale_factor_init = scale_factor_init
    self._arcface_margin = arcface_margin

  def init_classifiers(self, num_classes):
    """Define classifiers for training backbone and attention models."""
    logging.info('Initializing Delg backbone and attention models classifiers')
    backbone_classifier_func = self._create_backbone_classifier(num_classes)
    super(Delg, self).init_classifiers(
        num_classes,
        desc_classification=backbone_classifier_func)

  def _create_backbone_classifier(self, num_classes):
    """Define the classifier for training the backbone model."""
    logging.info('Creating cosine classifier')
    self.cosine_weights = tf.Variable(
        initial_value=tf.initializers.GlorotUniform()(
            shape=[self._embedding_layer_dim, num_classes]),
        name='cosine_weights',
        trainable=True)
    self.scale_factor = tf.Variable(self._scale_factor_init,
                                    name='scale_factor',
                                    trainable=False)
    classifier_func = functools.partial(cosine_classifier_logits,
                                        num_classes=num_classes,
                                        cosine_weights=self.cosine_weights,
                                        scale_factor=self.scale_factor,
                                        arcface_margin=self._arcface_margin)
    classifier_func.trainable_weights = [self.cosine_weights]
    return classifier_func


