import tensorflow as tf
# import tensorflow_probability as tfp

import os

from absl import logging

import pandas as pd
import glob

from tensorflow.keras import layers
import sys

from model import Delf, Delg
import matplotlib.pyplot as plt

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

DEBUG = False
using_delg = False

def create_model(num_classes):
  if using_delg:
    model = Delg(use_dim_reduction= True,)
  else:
    model = Delf(use_dim_reduction=True, )
  model.init_classifiers(num_classes)
  return model


## Data Prep
def data_prep():
  df = pd.read_csv("/scratch/as216/amur/reid_list_train.csv", header=None)
  df.columns = ["entity_id", "image_id"]

  df.sort_values("image_id", inplace=True)

  labels = list(df["entity_id"])
  image_ids = list((df["image_id"]))

  for filename in glob.glob("/scratch/as216/amur/train/*"):
    if filename[26:] not in image_ids:
      command = "rm " + filename
      os.system(command)

  next_mapping = 0
  labels_map = {}

  mapped_labels = []

  for elem in labels:
    if elem not in labels_map.keys():
      labels_map[elem] = next_mapping
      next_mapping += 1

    mapped_labels.append(labels_map[elem])

  return image_ids, mapped_labels, labels, labels_map

  return image_ids

def _learning_rate_schedule(global_step_value, max_iters, initial_lr):
  """Calculates learning_rate with linear decay.

  Args:
    global_step_value: int, global step.
    max_iters: int, maximum iterations.
    initial_lr: float, initial learning rate.

  Returns:
    lr: float, learning rate.
  """
  lr = initial_lr * (1.0 - global_step_value / max_iters)
  return lr

def _record_accuracy(metric, logits, labels):
  """Record accuracy given predicted logits and ground-truth labels."""
  softmax_probabilities = tf.keras.layers.Softmax()(logits)
  metric.update_state(labels, softmax_probabilities)



def rescale(image, label):
  image = tf.cast(image, tf.float32)
  image = (image / 255.0)
  return image, label

def augment(image_label, seed, img_height = 256, img_width = 512):
  image, label = image_label
  image, label = rescale(image, label)
  image = tf.image.resize_with_crop_or_pad(image, img_height + 6, img_width + 6)

  # Make a new seed.
  new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

  # Random crop back to the original size.
  image = tf.image.stateless_random_crop(image, size=[img_height, img_width, 3], seed=seed)

  # Random brightness.
  image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=new_seed)
  image = tf.clip_by_value(image, 0, 1)

  return image, label

def main():
  args = sys.argv[1:]

  # Setting GPUs up for use
  device_list = tf.config.list_physical_devices('GPU')
  gpus = [device_list[int(elem)] for elem in args]

  tf.config.set_visible_devices(gpus, 'GPU')

  # Setting up data
  image_ids, mapped_labels, labels, labels_map = data_prep()

  # Defining Constants
  data_dir = "/scratch/as216/amur/train/"
  logdir = "/home/as216/amur_delg/tf_implementation/log"
  image_size = 256
  initial_lr = 0.01
  batch_size = 32
  # max_iters = 50000
  max_iters = 2500
  num_eval_batches = 10
  save_interval = 500
  report_interval = 10
  eval_interval = 100
  num_classes = 107
  clip_val = tf.constant(10.0)

  if DEBUG:
    batch_size = 4
    max_iters = 100
    num_eval_batches = 1
    save_interval = 1
    report_interval = 10

  counter = tf.data.experimental.Counter()

  # Dataset Definition
  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels = mapped_labels,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(image_size, image_size*2),
    batch_size=batch_size).unbatch()

  val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels=mapped_labels,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(image_size, image_size*2),
    batch_size=batch_size).unbatch()

  AUTOTUNE = tf.data.AUTOTUNE

  data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("vertical"),
    layers.RandomRotation(1/72),
  ])

  train_ds = train_ds.repeat(count=10)

  train_ds = tf.data.Dataset.zip((train_ds, (counter, counter)))
  train_ds = (
    train_ds.shuffle(1000)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls = AUTOTUNE)
    .prefetch(AUTOTUNE)
    .repeat()
  )

  val_ds = (
    val_ds.shuffle(1000)
    .map(rescale, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
    .batch(batch_size)
    .repeat()
  )

  # Start Training
  strategy = tf.distribute.MirroredStrategy()

  print('Number of devices:', strategy.num_replicas_in_sync)

  if DEBUG:
    tf.config.run_functions_eagerly(True)

  train_dataset = strategy.experimental_distribute_dataset(train_ds)
  val_dataset = strategy.experimental_distribute_dataset(val_ds)

  train_iter = iter(train_dataset)
  validation_iter = iter(val_dataset)

  train_desc_acc = []
  train_attn_acc = []

  val_desc_acc = []
  val_attn_acc = []

  checkpoint_prefix = os.path.join(logdir, 'delf_tf2-ckpt')

  with strategy.scope():
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(labels, predictions):
      per_example_loss = loss_object(labels, predictions)
      return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)

    # Set up metrics.
    desc_validation_loss = tf.keras.metrics.Mean(name='desc_validation_loss')
    attn_validation_loss = tf.keras.metrics.Mean(name='attn_validation_loss')
    desc_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='desc_train_accuracy')
    attn_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='attn_train_accuracy')
    desc_validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='desc_validation_accuracy')
    attn_validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='attn_validation_accuracy')

    # Setup DELF model and optimizer.
    model = create_model(num_classes)

    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)

    # Setup checkpoint directory.
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
      checkpoint,
      checkpoint_prefix,
      max_to_keep=10,
      keep_checkpoint_every_n_hours=3)

    # Restores the checkpoint, if existing.
    # checkpoint.restore(manager.latest_checkpoint)

    # ------------------------------------------------------------
    # Train step to run on one GPU.
    def train_step(inputs):
      """Train one batch."""
      images, labels = inputs

      # Temporary workaround to avoid some corrupted labels.
      # labels = tf.clip_by_value(labels, 0, model.num_classes)


      def _backprop_loss(tape, loss, weights):
        """Backpropogate losses using clipped gradients.

        Args:
          tape: gradient tape.
          loss: scalar Tensor, loss value.
          weights: keras model weights.
        """
        gradients = tape.gradient(loss, weights)
        clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=clip_val)
        optimizer.apply_gradients(zip(clipped, weights))

      # Record gradients and loss through backbone.
      with tf.GradientTape() as gradient_tape:
        # Make a forward pass to calculate prelogits.
        (desc_prelogits, attn_prelogits, attn_scores, backbone_blocks,
         dim_expanded_features, _) = model.global_and_local_forward_pass(images)

        # Calculate global loss by applying the descriptor classifier.

        if using_delg:
          desc_logits = model.desc_classification(desc_prelogits, labels)
        else:
          desc_logits = model.desc_classification(desc_prelogits)
        desc_loss = compute_loss(labels, desc_logits)

        # Calculate attention loss by applying the attention block classifier.
        attn_logits = model.attn_classification(attn_prelogits)
        attn_loss = compute_loss(labels, attn_logits)

        # Calculate reconstruction loss between the attention prelogits and the
        # backbone.

        block3 = tf.stop_gradient(backbone_blocks['block3'])
        reconstruction_loss = tf.math.reduce_mean(
          tf.keras.losses.MSE(block3, dim_expanded_features))

        # reconstruction_loss = 0;

        # Cumulate global loss, attention loss and reconstruction loss.
        total_loss = (desc_loss + attn_loss + 10.0 * reconstruction_loss)

      # Perform backpropagation through the descriptor and attention layers
      # together. Note that this will increment the number of iterations of
      # "optimizer".
      _backprop_loss(gradient_tape, total_loss, model.trainable_weights)

      _record_accuracy(desc_train_accuracy, desc_logits, labels)
      _record_accuracy(attn_train_accuracy, attn_logits, labels)

      return desc_loss, attn_loss, reconstruction_loss

    # ------------------------------------------------------------
    def validation_step(inputs):
      """Validate one batch."""
      images, labels = inputs
      # labels = tf.clip_by_value(labels, 0, model.num_classes)

      # Get descriptor predictions.
      blocks = {}
      prelogits = model.backbone(
        images, intermediates_dict=blocks, training=False)

      if using_delg:
        logits = model.desc_classification(prelogits, labels)
      else:
        logits = model.desc_classification(prelogits, training=False)
      softmax_probabilities = tf.keras.layers.Softmax()(logits)

      validation_loss = loss_object(labels, logits)
      desc_validation_loss.update_state(validation_loss)
      desc_validation_accuracy.update_state(labels, softmax_probabilities)

      # Get attention predictions.
      block3 = blocks['block3']  # pytype: disable=key-error
      prelogits, _, _ = model.attention(block3, training=False)

      logits = model.attn_classification(prelogits, training=False)
      softmax_probabilities = tf.keras.layers.Softmax()(logits)

      validation_loss = loss_object(labels, logits)
      attn_validation_loss.update_state(validation_loss)
      attn_validation_accuracy.update_state(labels, softmax_probabilities)

      return desc_validation_accuracy.result(), attn_validation_accuracy.result(
      )

    # `run` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs):
      """Get the actual losses."""
      # Each (desc, attn) is a list of 3 losses - crossentropy, reg, total.
      desc_per_replica_loss, attn_per_replica_loss, recon_per_replica_loss = (
        strategy.run(train_step, args=(dataset_inputs,)))

      # Reduce over the replicas.
      desc_global_loss = strategy.reduce(
        tf.distribute.ReduceOp.SUM, desc_per_replica_loss, axis=None)
      attn_global_loss = strategy.reduce(
        tf.distribute.ReduceOp.SUM, attn_per_replica_loss, axis=None)
      recon_global_loss = strategy.reduce(
        tf.distribute.ReduceOp.SUM, recon_per_replica_loss, axis=None)

      return desc_global_loss, attn_global_loss, recon_global_loss

    @tf.function
    def distributed_validation_step(dataset_inputs):
      return strategy.run(validation_step, args=(dataset_inputs,))

    # ------------------------------------------------------------
    # *** TRAIN LOOP ***
    num_epochs = 10

    # for epoch in range(num_epochs):
    #
    #   print("Starting Epoch #" + str(epoch + 1))
    #
    #   train_iter = iter(train_dataset)
    #   validation_iter = iter(val_dataset)
    #
    #   global_step_value = optimizer.iterations.numpy()
    #   step_value = 0
    #
    #   # TODO(dananghel): try to load pretrained weights at backbone creation.
    #   # Load pretrained weights for ResNet50 trained on ImageNet.
    #   if (not global_step_value):
    #     input_batch = next(train_iter)
    #     _, _, _ = distributed_train_step(input_batch)
    #     model.backbone.restore_weights("/home/as216/amur_delg/tf_implementation/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
    #   else:
    #     logging.info('Skip loading ImageNet pretrained weights.')
    #
    #   model.backbone.log_weights()
    #
    #
    #   while step_value < max_iters:
    #     # input_batch : images(b, h, w, c), labels(b,).
    #     try:
    #       input_batch = next(train_iter)
    #     except tf.errors.OutOfRangeError:
    #       # Break if we run out of data in the dataset.
    #       logging.info('Stopping training at global step %d, no more data',
    #                    global_step_value)
    #       break
    #
    #     # Set learning rate and run the training step over num_gpu gpus.
    #     optimizer.learning_rate = _learning_rate_schedule(
    #       optimizer.iterations.numpy(), max_iters, initial_lr)
    #     desc_dist_loss, attn_dist_loss, recon_dist_loss = (
    #       distributed_train_step(input_batch))
    #
    #     # Step number, to be used for summary/logging.
    #     global_step = optimizer.iterations
    #     global_step_value = global_step.numpy()
    #
    #     template = ("Epoch {}, Desc Loss: {}, Attn Loss: {}, Recon Loss: {}, Desc Accuracy: {}, Attn Accuracy: {}")
    #     print(template.format(epoch + 1, desc_dist_loss, attn_dist_loss, recon_dist_loss, desc_train_accuracy.result() * 100, attn_train_accuracy.result() * 100))
    #
    #     # Print to console if running locally.
    #     if global_step_value % report_interval == 0:
    #       print(global_step.numpy())
    #       print('desc:', desc_dist_loss.numpy())
    #       print('attn:', attn_dist_loss.numpy())
    #
    #     # Validate once in {eval_interval*n, n \in N} steps.
    #     if step_value % eval_interval == 0:
    #       for i in range(num_eval_batches):
    #         try:
    #           validation_batch = next(validation_iter)
    #           desc_validation_result, attn_validation_result = (
    #             distributed_validation_step(validation_batch))
    #           print("\n------------------\n")
    #           print('Validation: desc:', desc_validation_result.numpy())
    #           print('          : attn:', attn_validation_result.numpy())
    #           print("\n------------------\n")
    #         except tf.errors.OutOfRangeError:
    #           logging.info('Stopping eval at batch %d, no more data', i)
    #           break
    #
    #       print("\n------------------\n")
    #       print('Validation: desc:', desc_validation_result.numpy())
    #       print('          : attn:', attn_validation_result.numpy())
    #       print("\n------------------\n")
    #
    #     # Save checkpoint once (each save_interval*n, n \in N) steps, or if
    #     # this is the last iteration.
    #     # TODO(andrearaujo): save only in one of the two ways. They are
    #     # identical, the only difference is that the manager adds some extra
    #     # prefixes and variables (eg, optimizer variables).
    #     if (step_value % save_interval == 0) or (step_value >= max_iters):
    #       manager.save(checkpoint_number=global_step_value)
    #
    #       file_path = '%s/delf_weights' % logdir
    #       model.save_weights(file_path, save_format='tf')
    #
    #     # Reset metrics for next step.
    #     desc_train_accuracy.reset_states()
    #     attn_train_accuracy.reset_states()
    #     desc_validation_loss.reset_states()
    #     attn_validation_loss.reset_states()
    #     desc_validation_accuracy.reset_states()
    #     attn_validation_accuracy.reset_states()
    #
    #     step_value += 1
    #
    #   logging.info('Finished training for %d steps.', max_iters)

    global_step_value = optimizer.iterations.numpy()

    # Load pretrained weights for ResNet50 trained on ImageNet.
    if (not global_step_value):
      input_batch = next(train_iter)
      _, _, _ = distributed_train_step(input_batch)
      model.backbone.restore_weights(
        "/home/as216/amur_delg/tf_implementation/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
    else:
      logging.info('Skip loading ImageNet pretrained weights.')

    model.backbone.log_weights()


    while global_step_value < max_iters:
      # input_batch : images(b, h, w, c), labels(b,).
      try:
        input_batch = next(train_iter)
      except tf.errors.OutOfRangeError:
        # Break if we run out of data in the dataset.
        logging.info('Stopping training at global step %d, no more data',
                     global_step_value)
        break

      # Set learning rate and run the training step over num_gpu gpus.
      optimizer.learning_rate = _learning_rate_schedule(
        optimizer.iterations.numpy(), max_iters, initial_lr)
      desc_dist_loss, attn_dist_loss, recon_dist_loss = (
        distributed_train_step(input_batch))

      # Step number, to be used for summary/logging.
      global_step = optimizer.iterations
      global_step_value = global_step.numpy()

      template = ("Global Step {}, Desc Loss: {}, Attn Loss: {}, Recon Loss: {}, Desc Accuracy: {}, Attn Accuracy: {}")

      # Print to console if running locally.
      if global_step_value % report_interval == 0:
        print(template.format(global_step_value, desc_dist_loss, attn_dist_loss, recon_dist_loss,
                              desc_train_accuracy.result() * 100, attn_train_accuracy.result() * 100))

      # Validate once in {eval_interval*n, n \in N} steps.
      if global_step_value % eval_interval == 0:
        for i in range(num_eval_batches):
          try:
            validation_batch = next(validation_iter)
            desc_validation_result, attn_validation_result = (
              distributed_validation_step(validation_batch))
          except tf.errors.OutOfRangeError:
            logging.info('Stopping eval at batch %d, no more data', i)
            break

        logging.info('\nValidation(%f)\n', global_step_value)
        logging.info(': desc: %f\n', desc_validation_result.numpy())
        logging.info(': attn: %f\n', attn_validation_result.numpy())
        # Print to console.

        print('Validation: desc:', desc_validation_result.numpy())
        print('          : attn:', attn_validation_result.numpy())

        train_desc_acc.append(desc_train_accuracy.result().numpy())
        train_attn_acc.append(attn_train_accuracy.result().numpy())
        val_desc_acc.append(desc_validation_result.numpy())
        val_attn_acc.append(attn_validation_result.numpy())

      # Save checkpoint once (each save_interval*n, n \in N) steps, or if
      # this is the last iteration.
      # TODO(andrearaujo): save only in one of the two ways. They are
      # identical, the only difference is that the manager adds some extra
      # prefixes and variables (eg, optimizer variables).
      if (global_step_value % save_interval == 0) or (global_step_value >= max_iters):
        save_path = manager.save(checkpoint_number=global_step_value)
        file_path = '%s/delf_weights' % logdir
        model.save_weights(file_path, save_format='tf')


      # Reset metrics for next step.
      desc_train_accuracy.reset_states()
      attn_train_accuracy.reset_states()
      desc_validation_loss.reset_states()
      attn_validation_loss.reset_states()
      desc_validation_accuracy.reset_states()
      attn_validation_accuracy.reset_states()

    logging.info('Finished training for %d steps.', max_iters)

    steps = range(1, len(train_desc_acc) + 1)

    plt.plot(steps, train_desc_acc, '--', label='Training')
    plt.plot(steps, val_desc_acc, '-g', label='Validation')
    plt.title('Descriptor accuracy')
    plt.legend()

    if using_delg:
      plt.savefig('delg_training_desc_acc_plot.png')
    else:
      plt.savefig('delf_training_desc_acc_plot.png')

    plt.figure()

    plt.plot(steps, train_attn_acc, '--', label='Training')
    plt.plot(steps, val_attn_acc, '-g', label='Validation')
    plt.title('Attention accuracy')
    plt.legend()

    plt.show()

    if using_delg:
      plt.savefig('delg_training_attn_acc_plot.png')
    else:
      plt.savefig('delf_training_attn_acc_plot.png')

if __name__ == "__main__":
    main()


