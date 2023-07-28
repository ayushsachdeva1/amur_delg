from model import Delf, Delg
import tensorflow as tf
import numpy as np
import sys

def delf_test_build():
  image_size = 321
  num_classes = 1000
  batch_size = 2
  input_shape = (batch_size, image_size, image_size, 3)

  model = Delf(block3_strides=True, name='DELF')
  model.init_classifiers(num_classes)

  images = tf.random.uniform(input_shape, minval=-1.0, maxval=1.0, seed=0)
  blocks = {}

  # Get global feature by pooling block4 features.
  desc_prelogits = model.backbone(
      images, intermediates_dict=blocks, training=False)
  desc_logits = model.desc_classification(desc_prelogits)

  np.testing.assert_array_equal(desc_prelogits.shape, (batch_size, 2048))
  np.testing.assert_array_equal(desc_logits.shape, (batch_size, num_classes))

  features = blocks['block3']
  attn_prelogits, _, _ = model.attention(features)
  attn_logits = model.attn_classification(attn_prelogits)

  np.testing.assert_array_equal(attn_prelogits.shape, (batch_size, 1024))
  np.testing.assert_array_equal(attn_logits.shape, (batch_size, num_classes))

def delf_test_training():
  image_size = 321
  num_classes = 1000
  batch_size = 2
  clip_val = 10.0
  input_shape = (batch_size, image_size, image_size, 3)

  model = Delf(block3_strides=True, name='DELF')
  model.init_classifiers(num_classes)

  optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

  images = tf.random.uniform(input_shape, minval=0.0, maxval=1.0, seed=0)
  labels = tf.random.uniform((batch_size,),
                              minval=0,
                              maxval=model.num_classes - 1,
                              dtype=tf.int64)

  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

  def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(
        per_example_loss, global_batch_size=batch_size)

  print(images.shape)
  print(labels)

  with tf.GradientTape() as gradient_tape:
    (desc_prelogits, attn_prelogits, _, _, _,
      _) = model.global_and_local_forward_pass(images)

    print(desc_prelogits.shape, attn_prelogits.shape)

    # Calculate global loss by applying the descriptor classifier.
    desc_logits = model.desc_classification(desc_prelogits)
    desc_loss = compute_loss(labels, desc_logits)
    # Calculate attention loss by applying the attention block classifier.
    attn_logits = model.attn_classification(attn_prelogits)
    attn_loss = compute_loss(labels, attn_logits)
    # Cumulate global loss and attention loss and backpropagate through the
    # descriptor layer and attention layer together.
    total_loss = desc_loss + attn_loss

    print(desc_loss, attn_loss)

  gradients = gradient_tape.gradient(total_loss, model.trainable_weights)
  clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=clip_val)
  optimizer.apply_gradients(zip(clipped, model.trainable_weights))

def delg_test_build():
  image_size = 321
  num_classes = 1000
  batch_size = 2
  input_shape = (batch_size, image_size, image_size, 3)

  model = Delg(
      block3_strides=True,
      use_dim_reduction=True)
  model.init_classifiers(num_classes)

  images = tf.random.uniform(input_shape, minval=-1.0, maxval=1.0, seed=0)
  labels = tf.random.uniform((batch_size,),
                              minval=0,
                              maxval=model.num_classes - 1,
                              dtype=tf.int64)
  blocks = {}

  desc_prelogits = model.backbone(
      images, intermediates_dict=blocks, training=False)
  desc_logits = model.desc_classification(desc_prelogits, labels)
  np.testing.assert_array_equal(desc_prelogits.shape, (batch_size, 2048))
  np.testing.assert_array_equal(desc_logits.shape, (batch_size, num_classes))

  features = blocks['block3']
  attn_prelogits, _, _ = model.attention(features)
  attn_logits = model.attn_classification(attn_prelogits)
  np.testing.assert_array_equal(attn_prelogits.shape, (batch_size, 1024))
  np.testing.assert_array_equal(attn_logits.shape, (batch_size, num_classes))


def delg_test_training():
  image_size = 321
  num_classes = 1000
  batch_size = 2
  clip_val = 10.0
  input_shape = (batch_size, image_size, image_size, 3)

  model = Delg(
    block3_strides=True,
    use_dim_reduction=True)
  model.init_classifiers(num_classes)

  optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

  images = tf.random.uniform(input_shape, minval=0.0, maxval=1.0, seed=0)
  labels = tf.random.uniform((batch_size,),
                             minval=0,
                             maxval=model.num_classes - 1,
                             dtype=tf.int64)

  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

  def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(
      per_example_loss, global_batch_size=batch_size)

  print(images.shape)
  print(labels)

  with tf.GradientTape() as gradient_tape:
    (desc_prelogits, attn_prelogits, _, backbone_blocks,
     dim_expanded_features, _) = model.global_and_local_forward_pass(images)
    # Calculate global loss by applying the descriptor classifier.
    desc_logits = model.desc_classification(desc_prelogits, labels)
    desc_loss = compute_loss(labels, desc_logits)
    # Calculate attention loss by applying the attention block classifier.
    attn_logits = model.attn_classification(attn_prelogits)
    attn_loss = compute_loss(labels, attn_logits)
    # Calculate reconstruction loss between the attention prelogits and the
    # backbone.
    block3 = tf.stop_gradient(backbone_blocks['block3'])
    reconstruction_loss = tf.math.reduce_mean(
      tf.keras.losses.MSE(block3, dim_expanded_features))
    # Cumulate global loss and attention loss and backpropagate through the
    # descriptor layer and attention layer together.
    total_loss = desc_loss + attn_loss + reconstruction_loss
    print(total_loss)

  gradients = gradient_tape.gradient(total_loss, model.trainable_weights)
  clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=clip_val)
  optimizer.apply_gradients(zip(clipped, model.trainable_weights))

def main():
  args = sys.argv[1:]

  device_list = tf.config.list_physical_devices('GPU')
  gpus = [device_list[int(elem)] for elem in args]

  tf.config.set_visible_devices(gpus, 'GPU')

  delf_test_build()
  delf_test_training()

  delg_test_build()
  delg_test_training()

if __name__ == "__main__":
    main()


