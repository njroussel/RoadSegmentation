"""

New version of the code for clear understanding of the method with better modularization.

"""

import os
import sys

from PIL import Image

import global_vars
import global_vars_pp
import logger
import prediction_helpers as pred_help
from tf_helpers import *

# Initialisation of some flags for tensor flow
# (In this case we declare the directory to store nets as we go)

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/mnist',
    """Directory where to write event logs and checkpoint.""")

FLAGS = tf.app.flags.FLAGS

SEED = global_vars_pp.SEED


def main(argv=None):
    # setup seeds
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    # output and input files
    output_run_log = 'runs.txt'
    train_data_filename = './predictions_training/'
    train_labels_filename = './training/groundtruth/'

    # TODO: faire un logger
    # arrays to store the different scores
    f1_validation_per_epoch = []
    f1_training_per_epoch = []
    loss_per_recording_step = []

    # File regex to load the images for training and testing
    FILE_REGEX = "satImage_%.3d"

    # Getting training images
    print("\nLoading images :")
    print("******************************************************************************")
    FILE_REGEX = "prediction_%d"
    prediction_images = img_help.read_binary_images(train_data_filename, global_vars_pp.TRAINING_SIZE, FILE_REGEX)
    FILE_REGEX = "satImage_%.3d"
    label_images = img_help.read_binary_images(train_labels_filename, global_vars_pp.TRAINING_SIZE, FILE_REGEX)

    prediction_images = img_help.quantize_binary_images(prediction_images, global_vars.IMG_PATCH_SIZE,
                                                        global_vars_pp.IMG_PATCH_SIZE)
    label_images = img_help.quantize_binary_images(label_images, global_vars.IMG_PATCH_SIZE,
                                                   global_vars_pp.IMG_PATCH_SIZE)

    #####################
    #####################
    #####################

    # Getting the data on which we are going to train
    data, labels = preparing_data(
        prediction_images, label_images, global_vars_pp.ROTATE_IMAGES, global_vars_pp.ROTATED_IMG,
        global_vars_pp.IMG_PATCH_SIZE, global_vars_pp.IMG_BORDER)

    # Seperating our data in three distinct sets (taining, validation, testing)
    # and normalization
    (train_set, valid_set, test_set, means, stds) = seperate_set(data, labels,
                                                                 global_vars_pp.VALIDATION_TRAIN_PERC,
                                                                 global_vars_pp.VALIDATION_VAL_PERC)

    # Balancing data
    train_set = img_help.balance_data(train_set[0], train_set[1])

    print("******************************************************************************")
    print("\nWe will train on", len(train_set[0]), "patches of size",
          str(global_vars_pp.IMG_TOTAL_SIZE) + "x" + str(global_vars_pp.IMG_TOTAL_SIZE))

    print("\nInitializing tensorflow graphs for training and validating")

    num_epochs = global_vars_pp.NUM_EPOCHS

    # Initialization of placeholders for data and labels
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(
            global_vars_pp.BATCH_SIZE, global_vars_pp.IMG_TOTAL_SIZE,
            global_vars_pp.IMG_TOTAL_SIZE, global_vars_pp.NUM_CHANNELS))

    train_label_node = tf.placeholder(
        tf.float32,
        shape=(global_vars_pp.BATCH_SIZE, global_vars_pp.NUM_LABELS))

    eval_data_node = tf.placeholder(
        tf.float32,
        shape=(None, global_vars_pp.IMG_TOTAL_SIZE,
               global_vars_pp.IMG_TOTAL_SIZE, global_vars_pp.NUM_CHANNELS))

    eval_label_node = tf.placeholder(
        tf.float32,
        shape=(None, global_vars_pp.NUM_LABELS))

    # Define the parameters of the convolutional layers
    conv_params, last_depth = params_conv_layers(
        global_vars_pp.CONV_ARCH, global_vars_pp.CONV_DEPTH, global_vars_pp.NUM_CHANNELS, SEED)

    pool_fact = 2 ** len(global_vars_pp.CONV_ARCH)

    if global_vars_pp.IMG_TOTAL_SIZE % pool_fact != 0:
        raise "not dividable by pool fact " + str(global_vars_pp.IMG_TOTAL_SIZE) + " / " + str(pool_fact)

    size = int(global_vars_pp.IMG_TOTAL_SIZE / pool_fact * global_vars_pp.IMG_TOTAL_SIZE / pool_fact * last_depth)

    fc_params = params_fc_layers(global_vars_pp.FC_ARCH, global_vars_pp.FC_DEPTH, size, global_vars_pp.NUM_LABELS, SEED)

    # Definition of the complete cnn model.
    def model(data, train=False):

        # convolution layers
        conv_end = init_conv_layers(global_vars_pp.CONV_ARCH, conv_params, data)

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        conv_end_shape = conv_end.get_shape().as_list()

        reshape = tf.reshape(
            conv_end,
            [-1, conv_end_shape[1] * conv_end_shape[2] * conv_end_shape[3]])

        out = init_fc_layers(global_vars_pp.FC_ARCH, fc_params, reshape, train, SEED)

        return out

    logits = model(train_data_node, True)

    # Computes the probability error for each prediction
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits, train_label_node))
    tf.summary.scalar('loss', loss)

    # L2 regularization for the fully connected parameters.
    regularizers = tf.nn.l2_loss(fc_params[0][0]) + tf.nn.l2_loss(fc_params[0][1])
    for params in fc_params[1:]:
        regularizers += tf.nn.l2_loss(params[0])
        regularizers += tf.nn.l2_loss(params[1])

    # Add the regularization term to the loss.
    loss += 5e-4 * (regularizers)

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)

    # Use adam optimizer as it optimises automatically the learning rate.
    adam_opt = tf.train.AdamOptimizer(global_vars_pp.LEARNING_RATE)
    optimizer = adam_opt.minimize(loss, global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction_graph = tf.nn.softmax(logits)
    # Compute predictions for validation and test
    correct_predictions_train_graph = tf.equal(
        tf.argmax(train_prediction_graph, 1), tf.argmax(train_label_node, 1))
    # Accuracy for training
    accuracy_train_graph = tf.reduce_mean(tf.cast(correct_predictions_train_graph, tf.float32))

    # Validation / Testing set predictions
    eval_predictions_graph = tf.nn.softmax(model(eval_data_node))
    # Compute predictions for validation and test
    eval_correct_predictions_graph = tf.equal(tf.argmax(eval_predictions_graph, 1), tf.argmax(eval_label_node, 1))
    # Accuracy computation
    eval_accuracy_graph = tf.reduce_mean(tf.cast(eval_correct_predictions_graph, tf.float32))

    # Will be used later when we need to compute the f1 score
    threshold_tf = tf.Variable(0, name="threshold_tf", dtype=tf.float32)

    # Index [0] corresponds to a road, which we will consider as positive therefore 1.
    pos_predictions_thresh_graph = tf.cast(tf.transpose(eval_predictions_graph)[0] > threshold_tf, tf.int64)
    # Here for the true labels we have the oposite -> 1 is background, road is 0 so we use argmin to reverse that
    true_predictions_graph = tf.argmin(eval_label_node, 1)
    # Here we have a boolean array with true values where instances of the prediction correspond to the labels.
    correct_predictions_thresh = tf.equal(pos_predictions_thresh_graph, true_predictions_graph)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # initialise all varibales operation
    init = tf.global_variables_initializer()

    s = tf.Session()

    if global_vars_pp.RESTORE_MODEL:
        # Restore variables from disk.
        saver.restore(s, FLAGS.train_dir + "/model.ckpt")
        print("Model restored.")
    else:
        # run initialisation of variables
        s.run(init)
        print('\nInitialized!')

        train_size = len(train_set[0])

        # Loop through training steps.
        print('\nTotal number of epochs for training :', num_epochs)
        print('Total number of steps for epoch :', int(train_size / global_vars_pp.BATCH_SIZE))
        print('Total number of steps :', num_epochs * int(train_size / global_vars_pp.BATCH_SIZE))
        print("\n")
        print("******************************************************************************")
        print("                                    Training")
        print("******************************************************************************")

        try:
            batch_size = global_vars_pp.BATCH_SIZE
            for epoch in range(num_epochs):
                print("\n******************************************************************************")
                print("training for epoch :", epoch + 1, "out of", num_epochs, "epochs")

                perm_idx = np.random.permutation(train_size)
                batch_bar = progressbar.ProgressBar(max_value=int(train_size / global_vars_pp.BATCH_SIZE))
                for step in range(int(train_size / global_vars_pp.BATCH_SIZE)):
                    batch_idx = perm_idx[step * batch_size: (step + 1) * batch_size]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_set[0][batch_idx]
                    batch_labels = train_set[1][batch_idx]

                    # This dictionary maps the batch data (as a np array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_label_node: batch_labels}

                    if step % global_vars_pp.RECORDING_STEP == 0:
                        _, l = s.run(
                            [optimizer, loss], feed_dict=feed_dict)

                        print("\ncomputing intermediate accuracy and loss at step", step)
                        print("computing train accuracy")
                        acc = batch_sum(s, eval_accuracy_graph, train_set, global_vars_pp.EVAL_BATCH_SIZE,
                                        eval_data_node,
                                        eval_label_node)
                        train_acc = acc / int(np.ceil(len(train_set[0]) / global_vars_pp.EVAL_BATCH_SIZE))
                        logger.append_log("Accuracy_training", train_acc)

                        print("computing validation accuracy")
                        acc = batch_sum(s, eval_accuracy_graph, valid_set, global_vars_pp.EVAL_BATCH_SIZE,
                                        eval_data_node,
                                        eval_label_node)
                        valid_acc = acc / int(np.ceil(len(valid_set[0]) / global_vars_pp.EVAL_BATCH_SIZE))
                        logger.append_log("Accuracy_validation", valid_acc)

                        print('\n%.2f' % (float(step) * global_vars_pp.BATCH_SIZE / train_size) + '% of Epoch ' + str(
                            epoch + 1))
                        print("loss :", l)
                        print("training set accuracy :", train_acc)
                        print("validation set accuracy :", valid_acc)

                        print("\nContinuing training steps")

                        # TODO: do a logging function
                        loss_per_recording_step.append(l)

                        sys.stdout.flush()
                    else:
                        # Run the graph and fetch some of the nodes.
                        batch_bar.update(step)
                        _, l, predictions_train = s.run(
                            [optimizer, loss, train_prediction_graph],
                            feed_dict=feed_dict)

                batch_bar.finish()

                # What do here ? nothing normally as done at beginning of each epoch
        except KeyboardInterrupt:
            print("Interrupted at epoch ", epoch + 1)
            pass

        logger.set_log("Epoch_stop", epoch + 1)

        print("\n******************************************************************************")
        print("Finished training")

        print("\nScoring on validation set")

        acc = batch_sum(s, eval_accuracy_graph, valid_set, global_vars_pp.EVAL_BATCH_SIZE, eval_data_node,
                        eval_label_node)
        accuracy = acc / int(np.ceil(len(valid_set[0]) / global_vars_pp.EVAL_BATCH_SIZE))
        logger.append_log("Accuracy_validation", accuracy)

        print("Accuracy rating is :", accuracy)

        print("\nScoring on testing set")

        acc = batch_sum(s, eval_accuracy_graph, test_set, global_vars_pp.EVAL_BATCH_SIZE, eval_data_node,
                        eval_label_node)
        accuracy = acc / int(np.ceil(len(test_set[0]) / global_vars_pp.EVAL_BATCH_SIZE))
        logger.set_log("Accuracy_test", accuracy)

        print("Accuracy rating is :", accuracy)

        print("\n******************************************************************************")
        print("Finding best f1_score with different thresholds")
        # Computing F1 score from predictions with different thresholds
        f1_scores = []
        threshs = np.linspace(0.3, 0.6, 10)

        for thresh in threshs:
            s.run(threshold_tf.assign(thresh))

            print("\nComputing F1-score with threshold :", thresh)

            f1_score = compute_f1_tf(s, pos_predictions_thresh_graph, correct_predictions_thresh, valid_set,
                                     global_vars_pp.EVAL_BATCH_SIZE, eval_data_node, eval_label_node)

            f1_scores.append(f1_score)
            logger.append_log("F1-score_validation", f1_score)

            print("F1-score :", f1_score)

        # Output test with best Threshold
        thresh = threshs[np.argmax(f1_scores)]

        # Test set f1_score

        s.run(threshold_tf.assign(thresh))

        print("\nTest set F1-score with best threshold :", thresh)

        f1_score = compute_f1_tf(s, pos_predictions_thresh_graph, correct_predictions_thresh, test_set,
                                 global_vars_pp.EVAL_BATCH_SIZE, eval_data_node, eval_label_node)

        logger.set_log("F1-score_test", f1_score)

        print("F1-score:", f1_score)

        if global_vars_pp.TRAIN_PREDICTIONS:
            ## Run on test set.
            print("\n******************************************************************************")
            print('Running on train set\n')
            FILE_REGEX = 'prediction_%d'
            test_data_filename = './test_predictions/'
            test_dir = 'predictions_training_post/'
            if not os.path.isdir(test_dir):
                os.mkdir(test_dir)
            for i in range(1, global_vars_pp.TRAINING_SIZE + 1):
                print('train prediction {}'.format(i))
                pimg = pred_help.get_prediction_image(test_data_filename, i, s, model, FILE_REGEX, means, stds,
                                                      global_vars_pp, thresh)
                pimg = \
                    img_help.quantize_binary_images([pimg], global_vars_pp.IMG_PATCH_SIZE, global_vars.IMG_PATCH_SIZE)[
                        0]
                pimg = pimg.reshape(pimg.shape[0], pimg.shape[1])
                pimg = img_help.img_float_to_uint8(pimg)

                Image.fromarray(pimg).save(test_dir + "prediction_" + str(i) + ".png")

        if global_vars_pp.TEST_PREDICTIONS:
            ## Run on test set.
            print("\n******************************************************************************")
            print('Running on test set\n')
            FILE_REGEX = 'prediction_%d'
            TEST_SIZE = 50
            test_data_filename = './test_predictions/'
            test_dir = 'test_predictions_post/'
            if not os.path.isdir(test_dir):
                os.mkdir(test_dir)
            for i in range(1, TEST_SIZE + 1):
                print('test prediction {}'.format(i))
                pimg = pred_help.get_prediction_image(test_data_filename, i, s, model, FILE_REGEX, means, stds,
                                                      global_vars_pp, thresh)
                pimg = \
                    img_help.quantize_binary_images([pimg], global_vars_pp.IMG_PATCH_SIZE, global_vars.IMG_PATCH_SIZE)[
                        0]

                pimg = pimg.reshape(pimg.shape[0], pimg.shape[1])
                pimg = img_help.img_float_to_uint8(pimg)
                Image.fromarray(pimg).save(test_dir + "prediction_" + str(i) + ".png")

        logger.save_log()


if __name__ == '__main__':
    tf.app.run()
