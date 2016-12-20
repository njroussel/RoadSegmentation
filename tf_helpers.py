import numpy as np
import progressbar
import tensorflow as tf

import image_helpers as img_help


def separate_set(data, labels, train_per, val_per):
    indices = np.random.permutation(data.shape[0])

    train_len = int(train_per * len(indices))
    valid_len = int(val_per * len(indices))

    train_idx = indices[:train_len]
    valid_idx = indices[train_len:train_len + valid_len]
    test_idx = indices[train_len + valid_len:]

    train_data = data[train_idx]
    train_labels = labels[train_idx]
    train_data, means, stds = img_help.standardize(train_data)

    valid_data = data[valid_idx]
    valid_labels = labels[valid_idx]
    valid_data, _, _ = img_help.standardize(valid_data, means=means, stds=stds)

    test_data = data[test_idx]
    test_labels = labels[test_idx]
    test_data, _, _ = img_help.standardize(test_data, means=means, stds=stds)

    return (
        (train_data, train_labels), (valid_data, valid_labels), (test_data, test_labels),
        means, stds)


def preparing_data(sat_images, label_images, rotate_image, nbr_rot, img_patch_size, img_border):
    # Adding rotated images
    if rotate_image:
        perm = np.random.choice(sat_images.shape[0], sat_images.shape[0])
        imgs = []
        labels = []

        for i in perm[0:nbr_rot]:
            angle = np.random.rand() * 360
            rot_img = img_help.rotate_image(sat_images[i], angle)
            rot_label = img_help.rotate_image(label_images[i], angle)
            imgs.append(rot_img)
            labels.append(rot_label)

        sat_images = np.append(sat_images, imgs, axis=0)
        label_images = np.append(label_images, labels, axis=0)

    print("\nWe have a total of", len(sat_images), "images for training.")

    # Extrcting patches from images
    data = img_help.extract_data(
        sat_images, img_patch_size, img_border)

    labels = img_help.extract_labels(label_images, img_patch_size)

    return data, labels


def batch_sum(s, func_sum, data_set, eval_batch_size, eval_data_node, eval_label_node):
    # TODO: put this in a function
    # Evaluating accuracy for EVAL_BATCH_SIZE parts of the validation set
    set_len = len(data_set[0])
    batch_nbr = int(set_len / eval_batch_size) + 1
    batch_idxs = np.array_split(range(set_len), batch_nbr)

    b_update = 0
    score_bar = progressbar.ProgressBar(max_value=len(batch_idxs)).start()

    acc = 0

    for batch_idx in batch_idxs:
        score_bar.update(b_update)
        b_update += 1

        if len(batch_idx) < eval_batch_size:
            batch_idx = range(set_len)[-eval_batch_size:]

        feed_dict = {eval_data_node: data_set[0][batch_idx],
                     eval_label_node: data_set[1][batch_idx]}

        acc += s.run(func_sum, feed_dict=feed_dict)

    score_bar.finish()
    return acc


def init_fc_layers(fc_arch, fc_params, prev_layer, dropout, seed=None):
    # convolution layers

    for i in range(fc_arch):
        prev_layer = tf.nn.relu(tf.matmul(prev_layer, fc_params[i][0]) + fc_params[i][1])
        if dropout:
            prev_layer = tf.nn.dropout(prev_layer, 0.8, seed=seed)

    fc_end = tf.matmul(prev_layer, fc_params[-1][0]) + fc_params[-1][1]

    return fc_end


def init_conv_layers(conv_arch, conv_params, prev_layer):
    # convolution layers

    for i, n_conv in enumerate(conv_arch):

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

    return prev_layer


def params_conv_layers(conv_arch, conv_depth, channels, seed=None):
    # init of conv parameters
    conv_params = [None] * len(conv_arch)

    # init origin depth
    prev_depth = channels
    for i, n_conv in enumerate(conv_arch):
        conv_params[i] = [None] * n_conv
        new_depth = conv_depth[i]
        for layer in range(n_conv):
            conv_weights = tf.Variable(
                tf.truncated_normal(
                    [3, 3, prev_depth, new_depth],  # 3x3 filter, depth augmenting by few steps.
                    stddev=0.1,
                    seed=seed))
            conv_biases = tf.Variable(tf.zeros([new_depth]))
            conv_params[i][layer] = (conv_weights, conv_biases)
            prev_depth = new_depth

    return conv_params, new_depth


def params_fc_layers(fc_arch, fc_depth, depth_conv, last_depth, seed=None):
    fc_param = []

    prev_depth = depth_conv
    for i in range(fc_arch):
        weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal(
                [prev_depth, fc_depth[i]],
                stddev=0.1,
                seed=seed))
        biases = tf.Variable(tf.constant(0.1, shape=[fc_depth[i]]))
        fc_param.append((weights, biases))
        prev_depth = fc_depth[i]

    last_weights = tf.Variable(
        tf.truncated_normal(
            [prev_depth, last_depth],
            stddev=0.1,
            seed=seed))
    last_biases = tf.Variable(tf.constant(0.1, shape=[last_depth]))
    fc_param.append((last_weights, last_biases))

    return fc_param


def init_cov_matrix_tf(predictions, correct_predictions):
    true_pred = tf.boolean_mask(predictions, correct_predictions)
    false_pred = tf.boolean_mask(predictions, tf.logical_not(correct_predictions))

    truePos = tf.reduce_sum(tf.cast(tf.equal(true_pred, 1), tf.float32))

    falsePos = tf.reduce_sum(tf.cast(tf.equal(false_pred, 1), tf.float32))

    trueNeg = tf.reduce_sum(tf.cast(tf.equal(true_pred, 0), tf.float32))

    falseNeg = tf.reduce_sum(tf.cast(tf.equal(false_pred, 0), tf.float32))

    return (truePos, falsePos, trueNeg, falseNeg)


def compute_f1_tf(s, predictions, correct_predictions, data_set, eval_batch_size, eval_data_node, eval_label_node):
    # Evaluating accuracy for EVAL_BATCH_SIZE parts of the validation set

    truePos, falsePos, trueNeg, falseNeg = init_cov_matrix_tf(predictions, correct_predictions)

    TP = 0
    FP = 0
    FN = 0

    set_len = len(data_set[0])
    batch_nbr = int(np.ceil(set_len / eval_batch_size))
    batch_idxs = np.array_split(range(set_len), batch_nbr)

    b_update = 0
    score_bar = progressbar.ProgressBar(max_value=len(batch_idxs))

    for batch_idx in batch_idxs:
        feed_dict = {eval_data_node: data_set[0][batch_idx],
                     eval_label_node: data_set[1][batch_idx]}

        truePos_res, falsePos_res, falseNeg_res = s.run([truePos, falsePos, falseNeg], feed_dict=feed_dict)

        TP += truePos_res
        FP += falsePos_res
        FN += falseNeg_res

        b_update += 1
        score_bar.update(b_update)

    if TP == 0:
        return 0

    if (TP + FP == 0):
        return 0

    if (FN + TP == 0):
        return 0

    precision = TP / (FP + TP)
    recall = TP / (FN + TP)

    score_bar.finish()
    return 2 * (precision * recall) / (precision + recall)
