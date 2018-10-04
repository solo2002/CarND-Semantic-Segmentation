#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time
import logging
import datetime as dt

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
  '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))


# # Check for a GPU
# if not tf.test.gpu_device_name():
#     warnings.warn('No GPU found. Please use a GPU to train your neural network.')
# else:
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
  """
  Load Pretrained VGG Model into TensorFlow.
  :param sess: TensorFlow Session
  :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
  :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
  """
  # TODO: Implement function
  #   Use tf.saved_model.loader.load to load the model and weights
  vgg_tag = 'vgg16'
  vgg_input_tensor_name = 'image_input:0'
  vgg_keep_prob_tensor_name = 'keep_prob:0'
  vgg_layer3_out_tensor_name = 'layer3_out:0'
  vgg_layer4_out_tensor_name = 'layer4_out:0'
  vgg_layer7_out_tensor_name = 'layer7_out:0'

  tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
  graph = tf.get_default_graph()
  image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
  keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
  layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
  layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
  layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

  return image_input, keep_prob, layer3_out, layer4_out, layer7_out


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
  """
  Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
  :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
  :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
  :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
  :param num_classes: Number of classes to classify
  :return: The Tensor for the last layer of output
  """
  # TODO: Implement function
  # layer 7 deconv
  layer7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1,  # kernal size
                                     padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

  # upsample
  layer7_upsampleBy2 = tf.layers.conv2d_transpose(layer7_conv_1x1, num_classes, 4, strides=(2, 2), padding='same',
                                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
  ########
  # layer 4 deconv
  layer4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1,  # kernal size
                                     padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

  # skip layer
  skip_layer47 = tf.add(layer7_upsampleBy2, layer4_conv_1x1)

  # upsample decon
  layer4_upsample = tf.layers.conv2d_transpose(skip_layer47, num_classes, 4, strides=(2, 2), padding='same',
                                               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
  ########
  layer3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1,  # kernal size
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                     padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

  # skip layer
  skip_layer34 = tf.add(layer4_upsample, layer3_conv_1x1)

  # upsample deconv
  layer3_upsample = tf.layers.conv2d_transpose(skip_layer34, num_classes, 16, strides=(8, 8), padding='same',
                                               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

  return layer3_upsample


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
  """
  Build the TensorFLow loss and optimizer operations.
  :param nn_last_layer: TF Tensor of the last layer in the neural network
  :param correct_label: TF Placeholder for the correct label image
  :param learning_rate: TF Placeholder for the learning rate
  :param num_classes: Number of classes to classify
  :return: Tuple of (logits, train_op, cross_entropy_loss)
  """
  # TODO: Implement function

  logits = tf.reshape(nn_last_layer, (-1, num_classes))
  correct_label = tf.reshape(correct_label, (-1, num_classes))

  # loss function
  cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))

  # Adam optimizer
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

  train_op = optimizer.minimize(cross_entropy_loss)

  return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def start_log(log_file_path):
  """
  Setup Logging system
  """
  if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)
  log_file = log_file_path + "/" + dt.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d %H_%M_%S') + ".log"
  file_handler = logging.FileHandler("{0}".format(log_file))
  log_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                    "%Y-%m-%d %H:%M:%S")
  file_handler.setFormatter(log_formatter)
  root_logger = logging.getLogger()
  root_logger.addHandler(file_handler)
  root_logger.setLevel(logging.INFO)
  root_logger.addHandler(logging.StreamHandler())


def run_time(seconds):
  """
  format time to hh:mm:ss
  :param seconds: number of seconds
  :return: a formatted time
  """
  m, s = divmod(seconds, 60)
  h, m = divmod(m, 60)
  return "%02d:%02d:%02d" % (h, m, s)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
  """
  Train neural network and print out the loss during training.
  :param sess: TF Session
  :param epochs: Number of epochs
  :param batch_size: Batch size
  :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
  :param train_op: TF Operation to train the neural network
  :param cross_entropy_loss: TF Tensor for the amount of loss
  :param input_image: TF Placeholder for input images
  :param correct_label: TF Placeholder for label images
  :param keep_prob: TF Placeholder for dropout keep probability
  :param learning_rate: TF Placeholder for learning rate
  """
  # TODO: Implement function

  start = time.time()
  #     start_log("training_results_log")

  sess.run(tf.global_variables_initializer())
  keep_prob_value = 0.5
  learning_rate_value = 0.0003  # 0.0001

  for epoch in range(epochs):
    print("****************************")
    print('Epoch: {}'.format(epoch + 1))
    total_loss = 0

    for image, label in get_batches_fn(batch_size):
      loss, _ = sess.run([cross_entropy_loss, train_op],
                         feed_dict={input_image: image, correct_label: label,
                                    keep_prob: keep_prob_value, learning_rate: learning_rate_value})

      total_loss += loss
      #         logging.info('Loss: = {:.2f}'.format(loss), '  Total Loss: = {:.2f}'.format(total_loss))
      #         logging.info("****************************")
      print("Loss: = {:.2f}".format(loss), "  Total Loss: = {:.2f}".format(total_loss))

    print()

  end = time.time()
  time_elapsed = end - start
  #     logging.info("Epoch: ", epochs, " batch: ", batch_size, " learning rate: ", learning_rate)
  print("Epoch: ", epochs, " batch size: ", batch_size, " learning rate: ", learning_rate_value)
  #     logging.info('Time elapsed (hh:mm:ss): ', run_time(time_elapsed))
  print('Time elapsed (hh:mm:ss): ', run_time(time_elapsed))


tests.test_train_nn(train_nn)


def run():
  num_classes = 2
  image_shape = (160, 576)
  data_dir = './data'
  runs_dir = './runs'
  tests.test_for_kitti_dataset(data_dir)

  # Download pretrained vgg model
  helper.maybe_download_pretrained_vgg(data_dir)

  # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
  # You'll need a GPU with at least 10 teraFLOPS to train on.
  #  https://www.cityscapes-dataset.com/

  with tf.Session() as sess:
    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')
    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    # OPTIONAL: Augment Images for better results
    #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

    # TODO: Build NN using load_vgg, layers, and optimize function
    input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
    layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

    # TODO: Train NN using the train_nn function
    epochs = 30
    batch_size = 5  # 10
    train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)
    # TODO: Save inference data using helper.save_inference_samples
    helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

    # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
  run()

