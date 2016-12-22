"""
Baseline for machine learning project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss

Credits: Aurelien Lucchi, ETH Zürich
"""

import sys
import time

from prediction_helpers import *

tf.app.flags.DEFINE_string('train_dir', '/tmp/mnist',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS


def validation(data, labels, s, model):
    pred = get_prediction_from_patches(data, s, model, EVAL_BATCH_SIZE, IMG_TOTAL_SIZE, NUM_CHANNELS, NUM_LABELS)
    f1_score = F1_score(pred, labels)
    return f1_score


def main(argv=None):  # pylint: disable=unused-argument
    np.random.seed(0xDEADBEEF)
    tf.set_random_seed(0xDEADBEEF)

    params_file_name = 'runs.txt'
    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'

    # Array containing all f1 score for validation ar each epoch (if enabled)
    f1_validation_per_epoch = []
    f1_training_per_epoch = []
    loss_per_recording_step = []

    # Extract it into np arrays.
    FILE_REGEX = "satImage_%.3d"

    sat_images, label_images = read_images(train_data_filename, train_labels_filename, TRAINING_SIZE,
                                           FILE_REGEX)

    perm_indices = np.random.permutation(range(sat_images[0:100].shape[0]))
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

    if TRAINING_SIZE > 100:
        # Add our images to training set only
        to_add = int(min(train_limit, TRAINING_SIZE - 100))
        excess_sat_images = sat_images[100:, :, :, :]
        excess_label_images = label_images[100:, :, :]

        excess_perm = np.random.permutation(range(excess_sat_images.shape[0]))

        train_data = np.append(train_data, excess_sat_images[excess_perm[:to_add]], axis=0)
        train_labels = np.append(train_labels, excess_label_images[excess_perm[:to_add]], axis=0)

    if ROTATE_IMAGES:
        for i in range(train_data.shape[0]):
            for j in range(ROTATION_AMOUNT):
                angle = np.random.rand() * 360
                rot_data = rotate_image(train_data[i], angle)
                rot_label = rotate_image(train_labels[i], angle)
                train_data = np.append(train_data, [rot_data], axis=0)
                train_labels = np.append(train_labels, [rot_label], axis=0)

    train_data = extract_data(train_data, IMG_PATCH_SIZE, IMG_BORDER)
    train_labels = extract_labels(train_labels, IMG_PATCH_SIZE)

    validation_data = extract_data(validation_data, IMG_PATCH_SIZE, IMG_BORDER)
    validation_labels = extract_labels(validation_labels, IMG_PATCH_SIZE)

    test_data = extract_data(test_data, IMG_PATCH_SIZE, IMG_BORDER)
    test_labels = extract_labels(test_labels, IMG_PATCH_SIZE)

    num_epochs = NUM_EPOCHS

    # Balancing is only performed on the training set.
    train_data, train_labels = balance_data(train_data, train_labels)
    train_size = train_labels.shape[0]

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
                    [FILTER_SIZE, FILTER_SIZE, prev_depth, new_depth],
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

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        last_layer_shape = last_layer.get_shape().as_list()
        reshape = tf.reshape(
            last_layer,
            [last_layer_shape[0], last_layer_shape[1] * last_layer_shape[2] * last_layer_shape[3]])

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        # if train:
        #   hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        out = tf.matmul(hidden, fc2_weights) + fc2_biases
        return out

    logits = model(train_data_node, True)  # BATCH_SIZE*NUM_LABELS
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

        try:
            for iepoch in range(num_epochs):

                # Permute training indices
                perm_indices = np.random.permutation(training_indices)

                for step in range(int(train_size / BATCH_SIZE)):

                    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # This dictionary maps the batch data (as a np array) to the
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

                        loss_per_recording_step.append(l)
                        print('loss = ', l)
                        print('%.2f' % (float(step) * BATCH_SIZE / train_size) + '% of Epoch ' + str(iepoch))

                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.
                        _, l, predictions = s.run(
                            [optimizer, loss, train_prediction],
                            feed_dict=feed_dict)

                if COMPUTE_VALIDATION_F1_SCORE_FOR_EACH_EPOCH:
                    f1_score_validation = validation(validation_data, validation_labels, s, model)
                    f1_validation_per_epoch.append(f1_score_validation)
                    f1_score_training = validation(train_data, train_labels, s, model)
                    f1_training_per_epoch.append(f1_score_training)
                    print('For epoch {} : F1 score on validation set = {}'.format(iepoch, f1_score_validation))
                    print('For epoch {} : F1 score on training set = {}'.format(iepoch, f1_score_training))

                # Save the variables to disk.
                if ENABLE_RECORDING:
                    save_path = saver.save(s, FLAGS.train_dir + "/model.ckpt")
                    print("Model saved in file: %s" % save_path)

        except KeyboardInterrupt:
            print("Interrupted at epoch ", iepoch + 1)
            pass

        if TRAIN_PREDICTIONS:
            print("Running prediction on training set")
            prediction_training_dir = "predictions_training/"
            if not os.path.isdir(prediction_training_dir):
                os.mkdir(prediction_training_dir)
            for i in range(1, TRAINING_SIZE + 1):
                print('prediction {}'.format(i))
                pimg = get_prediction_image(train_data_filename, i, s, model, FILE_REGEX, means, stds,
                                            IMG_PATCH_SIZE, IMG_BORDER, IMG_TOTAL_SIZE, NUM_CHANNELS,
                                            EVAL_BATCH_SIZE, NUM_LABELS)
                Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png")
                oimg = get_prediction_with_overlay(train_data_filename, i, s, model, FILE_REGEX, means, stds,
                                                   IMG_PATCH_SIZE, IMG_BORDER, IMG_TOTAL_SIZE, NUM_CHANNELS,
                                                   EVAL_BATCH_SIZE, NUM_LABELS)
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
                pimg = get_prediction_image(test_data_filename, i, s, model, FILE_REGEX, means, stds,
                                            IMG_PATCH_SIZE, IMG_BORDER, IMG_TOTAL_SIZE, NUM_CHANNELS,
                                            EVAL_BATCH_SIZE, NUM_LABELS)
                Image.fromarray(pimg).save(test_dir + "prediction_" + str(i) + ".png")
                oimg = get_prediction_with_overlay(test_data_filename, i, s, model, FILE_REGEX, means, stds,
                                                   IMG_PATCH_SIZE, IMG_BORDER, IMG_TOTAL_SIZE, NUM_CHANNELS,
                                                   EVAL_BATCH_SIZE, NUM_LABELS)
                oimg.save(test_dir + "overlay_" + str(i) + ".png")

    print("Begin validation")
    # Run Nico's code.
    validation_f1_score = validation(validation_data, validation_labels, s, model)
    print("Validation ends : F1 = ", validation_f1_score)

    print("Run on test set")
    # Run Nico's code.
    test_f1_score = validation(test_data, test_labels, s, model)
    print("Run on test ends : F1 = ", test_f1_score)

    s.close()

    # Writing all params and scores to files
    print("Write run informations to {} file".format(params_file_name))
    param_file = open(params_file_name, 'a')
    param_file.write("On {}:\n".format(time.strftime("%c")))
    param_file.write("NUM_CHANNELS            = {}\n".format(NUM_CHANNELS))
    param_file.write("PIXEL_DEPTH             = {}\n".format(PIXEL_DEPTH))
    param_file.write("NUM_LABELS              = {}\n".format(NUM_LABELS))
    param_file.write("TRAINING_SIZE           = {}\n".format(TRAINING_SIZE))
    param_file.write("SEED                    = {}\n".format(SEED))
    param_file.write("EVAL_BATCH_SIZE         = {}\n".format(EVAL_BATCH_SIZE))
    param_file.write("BATCH_SIZE              = {}\n".format(BATCH_SIZE))
    param_file.write("NUM_EPOCHS              = {}\n".format(NUM_EPOCHS))
    param_file.write("ROTATE_IMAGES           = {}\n".format(ROTATE_IMAGES))
    param_file.write("RESTORE_MODEL           = {}\n".format(RESTORE_MODEL))
    param_file.write("TRAIN_PREDICTIONS       = {}\n".format(TRAIN_PREDICTIONS))
    param_file.write("TEST_PREDICTIONS        = {}\n".format(TEST_PREDICTIONS))
    param_file.write("ENABLE_RECORDING        = {}\n".format(ENABLE_RECORDING))
    param_file.write("RECORDING_STEP          = {}\n".format(RECORDING_STEP))
    param_file.write("LEARNING_RATE           = {}\n".format(LEARNING_RATE))
    param_file.write("Last epoch              = {}\n".format(iepoch + 1))
    param_file.write("Validation F1 per epoch = {}\n".format(f1_validation_per_epoch))
    param_file.write("Training F1 per epoch   = {}\n".format(f1_training_per_epoch))
    param_file.write("Loss per recording step = {}\n".format(loss_per_recording_step))
    param_file.write("Validation F1 score     = {}\n".format(validation_f1_score))
    param_file.write("Test F1 score           = {}\n".format(test_f1_score))
    param_file.write("CONV_ARCH               = {}\n".format(CONV_ARCH))
    param_file.write("IMG_PATCH_SIZE          = {}\n".format(IMG_PATCH_SIZE))
    param_file.write("IMG_BORDER              = {}\n".format(IMG_BORDER))
    param_file.write("IMG_TOTAL_SIZE          = {}\n".format(IMG_TOTAL_SIZE))
    param_file.write("VALIDATION_TRAIN_PERC   = {}\n".format(VALIDATION_TRAIN_PERC))
    param_file.write("VALIDATION_VAL_PERC     = {}\n".format(VALIDATION_VAL_PERC))
    param_file.write("VALIDATION_TEST_PERC    = {}\n".format(VALIDATION_TEST_PERC))
    param_file.write("################################################################################\n")
    param_file.write("################################################################################\n\n")
    param_file.close()
    print("Done")


if __name__ == '__main__':
    tf.app.run()
