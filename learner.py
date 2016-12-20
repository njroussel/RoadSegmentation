"""

New version of the code for clear understanding of the method with better modularization.

"""

import os
import sys

import logger
from tf_helpers import *


def learn(sat_images, label_images, configuration, SEED, FLAGS, save_model_filepath):
    # Getting the data on which we are going to train

    data, labels = preparing_data(
        sat_images, label_images, configuration.ROTATE_IMAGES, configuration.ROTATED_IMG,
        configuration.IMG_PATCH_SIZE, configuration.IMG_BORDER)

    # Seperating our data in three distinct sets (taining, validation, testing)
    # and normalization
    (train_set, valid_set, test_set, means, stds) = separate_set(data, labels,
                                                                 configuration.VALIDATION_TRAIN_PERC,
                                                                 configuration.VALIDATION_VAL_PERC)
    # Balancing data
    train_set = img_help.balance_data(train_set[0], train_set[1])

    print("******************************************************************************")
    print("\nWe will train on", len(train_set[0]), "patches of size",
          str(configuration.IMG_TOTAL_SIZE) + "x" + str(configuration.IMG_TOTAL_SIZE))

    print("\nInitializing tensorflow graphs for training and validating")

    num_epochs = configuration.NUM_EPOCHS

    # Initialization of placeholders for data and labels
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(
            configuration.BATCH_SIZE, configuration.IMG_TOTAL_SIZE,
            configuration.IMG_TOTAL_SIZE, configuration.NUM_CHANNELS))

    train_label_node = tf.placeholder(
        tf.float32,
        shape=(configuration.BATCH_SIZE, configuration.NUM_LABELS))

    eval_data_node = tf.placeholder(
        tf.float32,
        shape=(None, configuration.IMG_TOTAL_SIZE,
               configuration.IMG_TOTAL_SIZE, configuration.NUM_CHANNELS))

    eval_label_node = tf.placeholder(
        tf.float32,
        shape=(None, configuration.NUM_LABELS))

    # Define the parameters of the convolutional layers
    conv_params, last_depth = params_conv_layers(
        configuration.CONV_ARCH, configuration.CONV_DEPTH, configuration.NUM_CHANNELS, SEED)

    pool_fact = 2 ** len(configuration.CONV_ARCH)

    if configuration.IMG_TOTAL_SIZE % pool_fact != 0:
        raise "not dividable by pool fact " + str(configuration.IMG_TOTAL_SIZE) + " / " + str(pool_fact)

    size = int(configuration.IMG_TOTAL_SIZE / pool_fact * configuration.IMG_TOTAL_SIZE / pool_fact * last_depth)

    fc_params = params_fc_layers(configuration.FC_ARCH, configuration.FC_DEPTH, size, configuration.NUM_LABELS, SEED)

    # Definition of the complete cnn model.
    def model(data, train=False):

        # convolution layers
        conv_end = init_conv_layers(configuration.CONV_ARCH, conv_params, data)

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        conv_end_shape = conv_end.get_shape().as_list()

        reshape = tf.reshape(
            conv_end,
            [-1, conv_end_shape[1] * conv_end_shape[2] * conv_end_shape[3]])

        out = init_fc_layers(configuration.FC_ARCH, fc_params, reshape, train, SEED)

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
    adam_opt = tf.train.AdamOptimizer(configuration.LEARNING_RATE)
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

    if configuration.RESTORE_MODEL:
        # Restore variables from disk.
        if not os.path.exists(save_model_filepath + ".index"):
            raise ValueError("model not found : " + save_model_filepath)
        saver.restore(s, save_model_filepath)
        print("Model restored from :", save_model_filepath)
    else:
        # run initialisation of variables
        s.run(init)
        print('\nInitialized!')

        train_size = len(train_set[0])

        # Loop through training steps.
        print('\nTotal number of epochs for training :', num_epochs)
        print('Total number of steps for epoch :', int(train_size / configuration.BATCH_SIZE))
        print('Total number of steps :', num_epochs * int(train_size / configuration.BATCH_SIZE))
        print("\n")
        print("******************************************************************************")
        print("                                    Training")
        print("******************************************************************************")

        try:
            batch_size = configuration.BATCH_SIZE
            for epoch in range(num_epochs):
                print("\n******************************************************************************")
                print("training for epoch :", epoch + 1, "out of", num_epochs, "epochs")

                perm_idx = np.random.permutation(train_size)
                batch_bar = progressbar.ProgressBar(max_value=int(train_size / configuration.BATCH_SIZE))
                for step in range(int(train_size / configuration.BATCH_SIZE)):
                    batch_idx = perm_idx[step * batch_size: (step + 1) * batch_size]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_set[0][batch_idx]
                    batch_labels = train_set[1][batch_idx]

                    # This dictionary maps the batch data (as a np array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_label_node: batch_labels}

                    if step % configuration.RECORDING_STEP == 0:
                        _, l = s.run(
                            [optimizer, loss], feed_dict=feed_dict)

                        print("\ncomputing intermediate accuracy and loss at step", step)
                        print("computing train accuracy")
                        acc = batch_sum(s, eval_accuracy_graph, train_set, configuration.EVAL_BATCH_SIZE, eval_data_node,
                                        eval_label_node)
                        train_acc = acc / int(np.ceil(len(train_set[0]) / configuration.EVAL_BATCH_SIZE))
                        logger.append_log("Accuracy_training", train_acc)

                        print("computing validation accuracy")
                        acc = batch_sum(s, eval_accuracy_graph, valid_set, configuration.EVAL_BATCH_SIZE, eval_data_node,
                                        eval_label_node)
                        valid_acc = acc / int(np.ceil(len(valid_set[0]) / configuration.EVAL_BATCH_SIZE))
                        logger.append_log("Accuracy_validation", valid_acc)

                        logger.append_log("Loss_taining", l)

                        print('\n%.2f' % (float(step) * configuration.BATCH_SIZE / train_size) + '% of Epoch ' + str(
                            epoch + 1))
                        print("loss :", l)

                        print("training set accuracy :", train_acc)
                        print("validation set accuracy :", valid_acc)

                        saver.save(s, FLAGS.train_dir + "/model.ckpt")

                        print("\nContinuing training steps")

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
            print("Restoring model from last evaluation")
            saver.restore(s, FLAGS.train_dir + "/model.ckpt")
            pass

        logger.set_log("Epoch_stop", epoch + 1)

    print("\n******************************************************************************")
    print("Finished training")

    print("\nScoring on validation set")

    acc = batch_sum(s, eval_accuracy_graph, valid_set, configuration.EVAL_BATCH_SIZE, eval_data_node, eval_label_node)
    accuracy = acc / int(np.ceil(len(valid_set[0]) / configuration.EVAL_BATCH_SIZE))
    logger.append_log("Accuracy_validation", accuracy)

    print("Accuracy rating is :", accuracy)

    print("\nScoring on testing set")

    acc = batch_sum(s, eval_accuracy_graph, test_set, configuration.EVAL_BATCH_SIZE, eval_data_node, eval_label_node)
    accuracy = acc / int(np.ceil(len(test_set[0]) / configuration.EVAL_BATCH_SIZE))
    logger.set_log("Accuracy_test", accuracy)

    print("Accuracy rating is :", accuracy)

    print("\n******************************************************************************")
    print("Finding best f1_score with different thresholds")
    # Computing F1 score from predictions with different thresholds
    thresh_start = 0
    thresh_end = 1
    thresh_steps = 10
    theta_thresh = configuration.THETA_THRESH

    diff_thresh = 1
    while (diff_thresh > theta_thresh):
        print("\nTesting for threshold between", thresh_start, "and", thresh_end)
        threshs = np.linspace(thresh_start, thresh_end, thresh_steps)
        f1_scores = []

        for thresh in threshs:
            s.run(threshold_tf.assign(thresh))

            print("\nComputing F1-score with threshold :", thresh)

            f1_score = compute_f1_tf(s, pos_predictions_thresh_graph, correct_predictions_thresh, valid_set,
                                     configuration.EVAL_BATCH_SIZE, eval_data_node, eval_label_node)

            f1_scores.append(f1_score)

            print("F1-score :", f1_score)

        # Output test with best Threshold
        logger.append_log("F1-score_validation", f1_scores)
        logger.append_log("F1-score_threshs_validation", threshs)
        idx_max_thresh = np.argmax(f1_scores)
        diff_thresh = f1_scores[idx_max_thresh] - f1_scores[0]
        thresh_start = threshs[max(idx_max_thresh - 1, 0)]
        thresh_end = threshs[min(idx_max_thresh + 1, thresh_steps - 1)]

        print("\nDifference :", diff_thresh)

    max_thresh = threshs[idx_max_thresh]

    print("Best threshold found with confidence", theta_thresh, ":", max_thresh)

    # Test set f1_score

    s.run(threshold_tf.assign(max_thresh))

    print("\nTest set F1-score with best threshold :", max_thresh)

    f1_score = compute_f1_tf(s, pos_predictions_thresh_graph, correct_predictions_thresh, test_set,
                             configuration.EVAL_BATCH_SIZE, eval_data_node, eval_label_node)

    logger.set_log("F1-score_test", f1_score)

    print("F1-score:", f1_score)

    if not configuration.RESTORE_MODEL:
        print("\nSaving our model")
        saver.save(s, save_model_filepath)

    logger.save_log()

    return s, model, means, stds, max_thresh
