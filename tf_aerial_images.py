"""
Baseline for machine learning project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss

Credits: Aurelien Lucchi, ETH Zürich
"""

import sys

from prediction_helpers import *

tf.app.flags.DEFINE_string('train_dir', '/tmp/mnist',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS


def main(argv=None):  # pylint: disable=unused-argument
    numpy.random.seed(0xDEADBEEF)

    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    test_data_filename = './test_set_images/'
    train_labels_filename = data_dir + 'groundtruth/'

    # Extract it into numpy arrays.
    FILE_REGEX = "satImage_%.3d"

    sat_images, label_images = read_images(train_data_filename, train_labels_filename, TRAINING_SIZE,
                                           FILE_REGEX)

    perm_indices = numpy.random.permutation(range(sat_images.shape[0]))
    train_limit = int(VALIDATION_TRAIN_PERC * len(perm_indices))
    val_limit = int(VALIDATION_VAL_PERC * len(perm_indices))

    train_data = sat_images[perm_indices[0:train_limit]]
    train_data, means, stds = standardize(train_data)
    train_labels = label_images[perm_indices[0:train_limit]]

    validation_data = sat_images[perm_indices[train_limit:train_limit + val_limit]]
    validation_data, _, _ = standardize(validation_data, means=means, stds=stds)
    validation_labels = label_images[perm_indices[train_limit:train_limit + val_limit]]

    test_data = sat_images[perm_indices[train_limit + val_limit:]]
    test_data, _, _ = standardize(test_data, means=means, stds=stds)
    test_labels = label_images[perm_indices[train_limit + val_limit:]]

    if ROTATE_IMAGES:
        for i in range(train_data.shape[0]):
            angle = numpy.random.rand() * 360
            rot_data = rotate_image(train_data[i], angle)
            rot_label = rotate_image(train_labels[i], angle)
            train_data = numpy.append(train_data, [rot_data], axis=0)
            train_labels = numpy.append(train_labels, [rot_label], axis=0)

    train_data = extract_data(train_data, border=IMG_BORDER)
    train_labels = extract_labels(train_labels)

    validation_data = extract_data(validation_data, border=IMG_BORDER)
    validation_labels = extract_labels(validation_labels)

    test_data = extract_data(test_data, border=IMG_BORDER)
    test_labels = extract_labels(test_labels)

    num_epochs = NUM_EPOCHS

    #train_data, train_labels = balance_data(train_data, train_labels)
    train_size = train_labels.shape[0]
    validation_data, validation_labels = balance_data(validation_data, validation_labels)
    test_data, test_labels = balance_data(test_data, test_labels)

    print("After evening size = {}".format(train_size))

    print("### validation sizes ###")
    print("train_data = {}".format(train_data.shape))
    print("train_labels = {}".format(train_labels.shape))
    print("validation_data = {}".format(validation_data.shape))
    print("validation_labels = {}".format(validation_labels.shape))
    print("test_data = {}".format(test_data.shape))
    print("test_labels = {}".format(test_labels.shape))
    print("######")

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_TOTAL_SIZE, IMG_TOTAL_SIZE, NUM_CHANNELS))

    train_labels_node = tf.placeholder(tf.float32,
                                       shape=(BATCH_SIZE, NUM_LABELS))

    train_all_data_node = tf.constant(train_data)

    # init weights
    conv_params = [None] * len(CONV_ARCH)

    prev_depth = NUM_CHANNELS
    for i, n_conv in enumerate(CONV_ARCH):
        conv_params[i] = [None] * n_conv
        new_depth = 32 * 2 ** i
        for layer in range(n_conv):
            conv_weights = tf.Variable(
                tf.truncated_normal(
                    [3, 3, prev_depth, new_depth],  # 3x3 filter, depth augmenting by few steps.
                    stddev=0.1,
                    seed=SEED))
            conv_biases = tf.Variable(tf.zeros([new_depth]))
            conv_params[i][layer] = (conv_weights, conv_biases)
            prev_depth = new_depth

    pool_fact = 2 ** len(CONV_ARCH)

    if IMG_TOTAL_SIZE % pool_fact != 0:
        raise "not dividable by pool fact " + str(IMG_TOTAL_SIZE) + " / " + str(pool_fact)

    size = int(IMG_TOTAL_SIZE / pool_fact * IMG_TOTAL_SIZE / pool_fact * new_depth)

    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal(
            [size, 512],
            stddev=0.1,
            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

    fc2_weights = tf.Variable(
        tf.truncated_normal(
            [512, NUM_LABELS],
            stddev=0.1,
            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""

        prev_layer = data
        for i, n_conv in enumerate(CONV_ARCH):

            for layer in range(n_conv):
                # 2D convolution
                conv = tf.nn.conv2d(
                    prev_layer,
                    conv_params[i][layer][0],
                    strides=[1, 1, 1, 1],
                    padding='SAME')

                # Bias and rectified linear non-linearity.
                relu = tf.nn.relu(tf.nn.bias_add(conv, conv_params[i][layer][1]))
                prev_layer = relu

            prev_layer = tf.nn.max_pool(
                prev_layer,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME')

        last_layer = prev_layer

        # TODO : Needs to be written differently for new layer system
        # Uncomment these lines to check the size of each layer
        # print 'data ' + str(data.get_shape())
        # print 'conv ' + str(conv.get_shape())
        # print 'relu ' + str(relu.get_shape())
        # print 'pool ' + str(pool.get_shape())
        # print 'pool2 ' + str(pool2.get_shape())


        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        last_layer_shape = last_layer.get_shape().as_list()
        reshape = tf.reshape(
            last_layer,
            [last_layer_shape[0], last_layer_shape[1] * last_layer_shape[2] * last_layer_shape[3]])

        print(last_layer_shape)
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        out = tf.matmul(hidden, fc2_weights) + fc2_biases

        # TODO : Needs to be written differently for new layer system
        # if train == True:
        #     summary_id = '_0'
        #     s_data = get_image_summary(data)
        #     filter_summary0 = tf.image_summary('summary_data' + summary_id, s_data)
        #     s_conv = get_image_summary(conv)
        #     filter_summary2 = tf.image_summary('summary_conv' + summary_id, s_conv)
        #     s_pool = get_image_summary(pool)
        #     filter_summary3 = tf.image_summary('summary_pool' + summary_id, s_pool)
        #     s_conv2 = get_image_summary(conv2)
        #     filter_summary4 = tf.image_summary('summary_conv2' + summary_id, s_conv2)
        #     s_pool2 = get_image_summary(pool2)
        #     filter_summary5 = tf.image_summary('summary_pool2' + summary_id, s_pool2)

        return out

    # Training computation: logits + cross-entropy loss.

    logits = model(train_data_node, True)  # BATCH_SIZE*NUM_LABELS

    # print('logits = ' + str(logits.get_shape()))
    # print('train_labels_node = ' + str(train_labels_node.get_shape()))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits, train_labels_node))
    tf.scalar_summary('loss', loss)

    # TODO : Needs to be written differently for new layer system
    # all_params_node = [conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights,
    #                    fc2_biases]
    # all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'fc1_weights', 'fc1_biases',
    #                     'fc2_weights', 'fc2_biases']
    # all_grads_node = tf.gradients(loss, all_params_node)
    # all_grad_norms_node = []
    # for i in range(0, len(all_grads_node)):
    #     norm_grad_i = tf.global_norm([all_grads_node[i]])
    #     all_grad_norms_node.append(norm_grad_i)
    #     tf.scalar_summary(all_params_names[i], norm_grad_i)

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)

    # Use adam optimizer as it optimises automatically the learning rate.
    adam_opt = tf.train.AdamOptimizer(LEARNING_RATE)
    optimizer = adam_opt.minimize(loss, global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # We'll compute them only once in a while by calling their {eval()} method.
    train_all_prediction = tf.nn.softmax(model(train_all_data_node))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    s = tf.Session()
    if RESTORE_MODEL:
        # Restore variables from disk.
        saver.restore(s, FLAGS.train_dir + "/model.ckpt")
        print("Model restored.")

    else:
        # Run all the initializers to prepare the trainable parameters.
        # tf.initialize_all_variables().run()
        s.run(init)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                graph=s.graph)
        print('Initialized!')
        # Loop through training steps.
        print('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))

        training_indices = range(train_size)

        for iepoch in range(num_epochs):

            # Permute training indices
            perm_indices = numpy.random.permutation(training_indices)

            for step in range(int(train_size / BATCH_SIZE)):

                offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                # Compute the offset of the current minibatch in the data.
                # Note that we could use better randomization across epochs.
                batch_data = train_data[batch_indices, :, :, :]
                batch_labels = train_labels[batch_indices]
                # This dictionary maps the batch data (as a numpy array) to the
                # node in the graph is should be fed to.
                feed_dict = {train_data_node: batch_data,
                             train_labels_node: batch_labels}

                if step % RECORDING_STEP == 0:

                    summary_str, _, l, predictions = s.run(
                        [summary_op, optimizer, loss, train_prediction],
                        feed_dict=feed_dict)
                    # summary_str = s.run(summary_op, feed_dict=feed_dict)
                    if ENABLE_RECORDING:
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                    # print_predictions(predictions, batch_labels)

                    print('%.2f' % (float(step) * BATCH_SIZE / train_size) + '% of Epoch ' + str(iepoch))
                    print('Minibatch loss: %.3f' % (l))
                    print('Minibatch error: %.1f%%' % error_rate(predictions,
                                                                 batch_labels))
                    print('Minibatch F1 score: %.3f' % F1_score(predictions,
                                                                batch_labels))

                    sys.stdout.flush()
                else:
                    # Run the graph and fetch some of the nodes.
                    _, l, predictions = s.run(
                        [optimizer, loss, train_prediction],
                        feed_dict=feed_dict)

            # Save the variables to disk.
            if ENABLE_RECORDING:
                save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                print("Model saved in file: %s" % save_path)

        if TRAIN_PREDICTIONS:
            print("Running prediction on training set")
            prediction_training_dir = "predictions_training/"
            if not os.path.isdir(prediction_training_dir):
                os.mkdir(prediction_training_dir)
            for i in range(1, TRAINING_SIZE + 1):
                print('prediction {}'.format(i))
                pimg = get_prediction_with_groundtruth(train_data_filename, i, s, model, FILE_REGEX, means, stds)
                Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
                oimg = get_prediction_with_overlay(train_data_filename, i, s, model, FILE_REGEX, means, stds)
                oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")

        if TEST_PREDICTIONS:
            ## Run on test set.
            print('Running on test set.')
            FILE_REGEX = 'test_%d'
            TEST_SIZE = 50
            test_data_filename = './test_set_images/'
            test_dir = 'test_predictions/'
            if not os.path.isdir(test_dir):
                os.mkdir(test_dir)
            for i in range(1, TEST_SIZE + 1):
                print('test prediction {}'.format(i))
                pimg = get_prediction_with_groundtruth(test_data_filename, i, s, model, FILE_REGEX, means, stds)
                Image.fromarray(pimg).save(test_dir + "prediction_" + str(i) + ".png")
                oimg = get_prediction_with_overlay(test_data_filename, i, s, model, FILE_REGEX, means, stds)
                oimg.save(test_dir + "overlay_" + str(i) + ".png")

    print("Begin validation")
    # Run Nico's code.
    pred = get_prediction_from_patches(validation_data, s, model)
    f1_score = F1_score(pred, validation_labels)
    print("Validation ends : F1 = ", f1_score)

    print("Run on test set")
    # Run Nico's code.
    pred = get_prediction_from_patches(test_data, s, model)
    f1_score = F1_score(pred, test_labels)
    print("Run on test ends : F1 = ", f1_score)

    s.close()


if __name__ == '__main__':
    tf.app.run()
