"""

New version of the code for clear understanding of the method with better modularization.

"""

import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os
import progressbar

import prediction_helpers as pred_help
import image_helpers as img_help
import global_vars
from tf_helpers import *

# Initialisation of some flags for tensor flow
# (In this case we declare the directory to store nets as we go)

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/mnist',
    """Directory where to write event logs and checkpoint.""")

FLAGS = tf.app.flags.FLAGS

SEED = global_vars.SEED

def main(argv=None):
    # setup seeds
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    # output and input files
    output_run_log = 'runs.txt'
    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'

    # TODO: faire un logger
    # arrays to store the different scores
    f1_validation_per_epoch = []
    f1_training_per_epoch = []
    loss_per_recording_step = []
    
    # File regex to load the images for training and testing
    FILE_REGEX = "satImage_%.3d"
    
    # Getting training images
    sat_images, label_images = img_help.read_images(
        train_data_filename, train_labels_filename, 
        global_vars.TRAINING_SIZE, FILE_REGEX)

    # Getting the data on ehich we are going to train
    data, labels = preparing_data(
        sat_images, label_images, global_vars.ROTATE_IMAGES, global_vars.ROTATED_IMG,
        global_vars.IMG_PATCH_SIZE, global_vars.IMG_BORDER)

    # Seperating our data in three distinct sets (taining, validation, testing)
    # and normalization
    (train_set, valid_set, test_set, means, stds) = seperate_set(data, labels, 
        global_vars.VALIDATION_TRAIN_PERC, global_vars.VALIDATION_VAL_PERC)

    # Balancing data
    train_set = img_help.balance_data(train_set[0], train_set[1])

    print("We will train on", len(train_set[0]), "patches of size", 
        str(global_vars.IMG_TOTAL_SIZE)+ "x" + str(global_vars.IMG_TOTAL_SIZE))

    num_epochs = global_vars.NUM_EPOCHS

    # Initialization of placeholders for data and labels
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(
            global_vars.BATCH_SIZE, global_vars.IMG_TOTAL_SIZE, 
            global_vars.IMG_TOTAL_SIZE, global_vars.NUM_CHANNELS))

    train_label_node = tf.placeholder(
        tf.float32,
        shape=(global_vars.BATCH_SIZE, global_vars.NUM_LABELS))

    eval_data_node = tf.placeholder(
        tf.float32,
        shape=(None, global_vars.IMG_TOTAL_SIZE, 
            global_vars.IMG_TOTAL_SIZE, global_vars.NUM_CHANNELS))

    eval_label_node = tf.placeholder(
        tf.float32,
        shape=(None, global_vars.NUM_LABELS))

    # Define the parameters of the convolutional layers
    conv_params, last_depth = params_conv_layers(
        global_vars.CONV_ARCH, global_vars.CONV_DEPTH, global_vars.NUM_CHANNELS, SEED)

    pool_fact = 2 ** len(global_vars.CONV_ARCH)

    if global_vars.IMG_TOTAL_SIZE % pool_fact != 0:
        raise "not dividable by pool fact " + str(global_vars.IMG_TOTAL_SIZE) + " / " + str(pool_fact)

    size = int(global_vars.IMG_TOTAL_SIZE / pool_fact * global_vars.IMG_TOTAL_SIZE / pool_fact * last_depth)

    fc_params = params_fc_layers(global_vars.FC_ARCH, global_vars.FC_DEPTH, size, global_vars.NUM_LABELS, SEED)

    # Definition of the complete cnn model.
    def model(data, train=False):

        # convolution layers
        conv_end = init_conv_layers(global_vars.CONV_ARCH, conv_params, data)

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        conv_end_shape = conv_end.get_shape().as_list()

        reshape = tf.reshape(
            conv_end,
            [-1, conv_end_shape[1] * conv_end_shape[2] * conv_end_shape[3]])

        out = init_fc_layers(global_vars.FC_ARCH, fc_params, reshape, train, SEED)

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
    adam_opt = tf.train.AdamOptimizer(global_vars.LEARNING_RATE)
    optimizer = adam_opt.minimize(loss, global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction_graph = tf.nn.softmax(logits)
    # Compute predictions for validation and test
    correct_predictions_train_graph = tf.equal(
        tf.argmax(train_prediction_graph,1), tf.argmax(train_label_node,1))
    # Accuracy for training
    accuracy_train_graph = tf.reduce_mean(tf.cast(correct_predictions_train_graph, tf.float32))

    # Validation / Testing set predictions
    eval_predictions_graph = tf.nn.softmax(model(eval_data_node))
    # Compute predictions for validation and test
    eval_correct_predictions_graph = tf.equal(tf.argmax(eval_predictions_graph,1), tf.argmax(eval_label_node,1))
    # Accuracy for test as a sum, as we will have to do a mean by patch
    eval_accuracy_graph = tf.reduce_mean(tf.cast(eval_correct_predictions_graph, tf.float32))

    # Will be used later when we need to compute the f1 score
    threshold_tf = tf.Variable(0, name="threshold_tf", dtype=tf.float32)

    # Index [0] corresponds to a road, which we will consider as positive therefore 1.
    pos_predictions_thresh_graph = tf.cast(tf.transpose(eval_predictions_graph)[0] > threshold_tf, tf.int64)
    # Here for the true labels we have the oposite -> 1 is background, road is 0 so we use argmin to reverse that
    true_predictions_graph = tf.argmin(eval_label_node,1)
    # Here we have a boolean array with true values where instances of the prediction correspond to the labels.
    correct_predictions_thresh = tf.equal(pos_predictions_thresh_graph, true_predictions_graph)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # initialise all varibales operation
    init = tf.global_variables_initializer()

    s = tf.Session()

    if global_vars.RESTORE_MODEL:
        # Restore variables from disk.
        saver.restore(s, FLAGS.train_dir + "/model.ckpt")
        print("Model restored.")
    else:
        # run initialisation of variables
        s.run(init)
        print('Initialized!')

        # Loop through training steps.
        print('Total number of iterations : ' + str(int(num_epochs * len(train_set[0]) / global_vars.BATCH_SIZE)))

        train_size = len(train_set[0])
        epoch_bar = progressbar.ProgressBar(max_value=num_epochs)

        try:
            batch_size = global_vars.BATCH_SIZE
            for epoch in range(num_epochs):
                print("training for epoch", epoch)
                epoch_bar.update(epoch)
                perm_idx = np.random.permutation(train_size)
                batch_bar = progressbar.ProgressBar(max_value=int(train_size / global_vars.BATCH_SIZE))
                for step in range(int(train_size / global_vars.BATCH_SIZE)):
                    batch_idx = perm_idx[step * batch_size : (step+1) * batch_size]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_set[0][batch_idx]
                    batch_labels = train_set[1][batch_idx]

                    # This dictionary maps the batch data (as a np array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_label_node: batch_labels}
                    
                    if step % global_vars.RECORDING_STEP == 0:
                        _, train_acc, l = s.run(
                            [optimizer, accuracy_train_graph, loss], feed_dict=feed_dict)
                        
                        print("computing intermediate accuracy")
                        acc = batch_sum(s, eval_accuracy_graph, valid_set, global_vars.EVAL_BATCH_SIZE, eval_data_node, eval_label_node)
                        valid_acc = acc / int(np.ceil(len(valid_set[0]) / global_vars.EVAL_BATCH_SIZE))

                        print('%.2f' % (float(step) * global_vars.BATCH_SIZE / train_size) + '% of Epoch ' + str(epoch + 1))
                        print("loss :",l)
                        print("training batch set accuracy :", train_acc)
                        print("validation set accuracy :", valid_acc)

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

        epoch_bar.finish()

        print("Scoring on validation set")

        acc = batch_sum(s, eval_accuracy_graph, valid_set, global_vars.EVAL_BATCH_SIZE, eval_data_node, eval_label_node)
        accuracy = acc / int(np.ceil(len(valid_set[0]) / global_vars.EVAL_BATCH_SIZE))

        print("Accuracy rating is :", accuracy)

        print("Scoring on testing set")

        acc = batch_sum(s, eval_accuracy_graph, test_set, global_vars.EVAL_BATCH_SIZE, eval_data_node, eval_label_node)
        accuracy = acc / int(np.ceil(len(test_set[0]) / global_vars.EVAL_BATCH_SIZE))

        print("Accuracy rating is :", accuracy)

        # Computing F1 score from predictions with different thresholds
        f1_scores = []
        threshs = np.linspace(0.3, 0.6, 10)

        for thresh in threshs:
            s.run(threshold_tf.assign(thresh))

            feed_dict = {eval_data_node: valid_set[0][:64],
                eval_label_node: valid_set[1][:64]}

            print("this is the predictions probabilities")
            print(s.run(pos_predictions_thresh_graph, feed_dict=feed_dict))
            print("this is the true predictions")
            print(s.run(tf.argmax(eval_label_node,1), feed_dict=feed_dict))


            print("Threshold :",thresh)

            f1_score = compute_f1_tf(s, pos_predictions_thresh_graph, correct_predictions_thresh, valid_set, 
                global_vars.EVAL_BATCH_SIZE, eval_data_node, eval_label_node)

            f1_scores.append(f1_score)

            print("F1 score :",f1_score)

        # Output test with best Threshold
        thresh = threshs[np.argmax(f1_scores)]

        # Test set f1_score

        s.run(threshold_tf.assign(thresh))

        print("Threshold :", thresh)

        f1_score = compute_f1_tf(s, pos_predictions_thresh_graph, correct_predictions_thresh, test_set, 
            global_vars.EVAL_BATCH_SIZE, eval_data_node, eval_label_node)

        print("Best F1 score for test set :", f1_score)

        if global_vars.TEST_PREDICTIONS:
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
                pimg = pred_help.get_prediction_image(test_data_filename, i, s, model, FILE_REGEX, means, stds,
                                            global_vars, thresh)
                Image.fromarray(pimg).save(test_dir + "prediction_" + str(i) + ".png")
                oimg = pred_help.get_prediction_with_overlay(test_data_filename, i, s, model, FILE_REGEX, means, stds,
                                                   global_vars, thresh)
                oimg.save(test_dir + "overlay_" + str(i) + ".png")

if __name__ == '__main__':
    tf.app.run()
    

    