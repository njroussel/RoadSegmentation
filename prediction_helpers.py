import tensorflow as tf

from image_helpers import *


def get_image_summary(img, idx=0):
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    img_w = img.get_shape().as_list()[1]
    img_h = img.get_shape().as_list()[2]
    min_value = tf.reduce_min(V)
    V = V - min_value
    max_value = tf.reduce_max(V)
    V = V / (max_value * PIXEL_DEPTH)
    V = tf.reshape(V, (img_w, img_h, 1))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, (-1, img_w, img_h, 1))
    return V


# Make an image summary for 3d tensor image with index idx
def get_image_summary_3d(img):
    V = tf.slice(img, (0, 0, 0), (1, -1, -1))
    img_w = img.get_shape().as_list()[1]
    img_h = img.get_shape().as_list()[2]
    V = tf.reshape(V, (img_w, img_h, 1))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, (-1, img_w, img_h, 1))
    return V


def get_prediction_image(filename, image_idx, s, model, file_regex, means, stds, global_vars, thresh):
    imageid = file_regex % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)
    tmp = np.array(img)
    if len(tmp.shape) == 2:
        imgs = quantize_binary_images([img], IMG_PATCH_SIZE, PP_IMG_PATCH_SIZE)
        img = imgs[0]
        img = img.reshape(img.shape[0], img.shape[1], 1)

    img_prediction = get_prediction(img, s, model, means, stds, global_vars, thresh)

    return img_float_to_uint8(img_prediction)


# Helper function to run multiple baches instead of one big dataset (crashes most GPU)
def prediction_in_batches(data, sess, eval_prediction, data_node, eval_batch_size, num_labels):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < eval_batch_size:
        raise ValueError("batch size for evals larger than dataset: %d" % size)

    predictions = np.ndarray(shape=(size, num_labels), dtype=np.float32)

    for begin in range(0, size, eval_batch_size):
        end = begin + eval_batch_size
        if end <= size:
            predictions[begin:end, :] = sess.run(
                eval_prediction,
                feed_dict={data_node: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(
                eval_prediction,
                feed_dict={data_node: data[-eval_batch_size:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


# Get prediction for given input image
def get_prediction(img, s, model, means, stds, global_vars, thresh):
    data = np.asarray(
        img_crop(img, global_vars.IMG_PATCH_SIZE, global_vars.IMG_PATCH_SIZE, border=global_vars.IMG_BORDER))

    if len(data.shape) < 4:
        data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

    data, _, _ = standardize(data, means, stds)

    data_node = tf.placeholder(
        tf.float32,
        shape=(
            global_vars.EVAL_BATCH_SIZE,
            global_vars.IMG_TOTAL_SIZE,
            global_vars.IMG_TOTAL_SIZE,
            global_vars.NUM_CHANNELS))

    output = tf.nn.softmax(model(data_node))
    output_prediction = prediction_in_batches(
        data, s, output, data_node, global_vars.EVAL_BATCH_SIZE, global_vars.NUM_LABELS)

    img_prediction = label_to_img(
        img.shape[0], img.shape[1], global_vars.IMG_PATCH_SIZE, global_vars.IMG_PATCH_SIZE,
        output_prediction, thresh)

    return img_prediction


def get_prediction_from_patches(patches, s, model, eval_batch_size, img_total_size, num_channels, num_labels):
    data = np.asarray(patches)

    data_node = tf.placeholder(
        tf.float32,
        shape=(eval_batch_size, img_total_size, img_total_size, num_channels))

    output = tf.nn.softmax(model(data_node))
    output_prediction = prediction_in_batches(data, s, output, data_node, eval_batch_size, num_labels)
    return output_prediction


# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(filename, image_idx, s, model, file_regex, means, stds, global_vars, thresh):
    imageid = file_regex % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)
    tmp = np.array(img)
    if len(tmp.shape) == 3:
        img = img[:, :, :3]

    img_prediction = get_prediction(img, s, model, means, stds, global_vars, thresh)

    oimg = make_img_overlay(img, img_prediction)

    return oimg


def F1_score(predictions, labels):
    valid_index = np.argmax(predictions, 1) == np.argmax(labels, 1)
    valid_prediction = predictions[valid_index]
    false_prediction = predictions[~valid_index]

    # True positive
    TP = np.sum(np.argmax(valid_prediction, axis=1) == 1)
    # True negative
    TN = np.sum(np.argmax(valid_prediction, axis=1) == 0)
    # False positive
    FP = np.sum(np.argmax(false_prediction, axis=1) == 1)
    # False negative
    FN = np.sum(np.argmax(false_prediction, axis=1) == 0)

    precision = TP / (FP + TP)
    recall = TP / (FN + TP)

    return 2 * (precision * recall) / (precision + recall)
