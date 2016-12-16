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


def get_prediction_image(filename, image_idx, s, model, file_regex, means, stds, img_patch_size, img_border,
                         img_total_size, num_channels, eval_batch_size, num_labels):
    imageid = file_regex % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)
    tmp = np.array(img)
    if len(tmp.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)

    img_prediction = get_prediction(img, s, model, means, stds, img_patch_size, img_border, img_total_size,
                                    num_channels, eval_batch_size, num_labels)
    return img_float_to_uint8(img_prediction)


# Helper function to run multiple baches instead of one big dataset (crashes most GPU)
def eval_in_batches(data, sess, eval_prediction, data_node, eval_batch_size, num_labels):
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
def get_prediction(img, s, model, means, stds, img_patch_size, img_border, img_total_size, num_channels,
                   eval_batch_size, num_labels):
    data = np.asarray(img_crop(img, img_patch_size, img_patch_size, border=img_border))
    if len(data.shape) < 4:
        data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

    data, _, _ = standardize(data, means, stds)

    data_node = tf.placeholder(
        tf.float32,
        shape=(eval_batch_size, img_total_size, img_total_size, num_channels))


    output = tf.nn.softmax(model(data_node))
    output_prediction = eval_in_batches(data, s, output, data_node, eval_batch_size, num_labels)

    img_prediction = label_to_img(img.shape[0], img.shape[1], img_patch_size, img_patch_size, output_prediction)
    return img_prediction


def get_prediction_from_patches(patches, s, model, eval_batch_size, img_total_size, num_channels, num_labels):
    data = np.asarray(patches)

    data_node = tf.placeholder(
        tf.float32,
        shape=(eval_batch_size, img_total_size, img_total_size, num_channels))

    output = tf.nn.softmax(model(data_node))
    output_prediction = eval_in_batches(data, s, output, data_node, eval_batch_size, num_labels)
    return output_prediction


# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(filename, image_idx, s, model, file_regex, means, stds, img_patch_size, img_border,
                                img_total_size, num_channels, eval_batch_size, num_labels):
    imageid = file_regex % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)
    tmp = np.array(img)
    if len(tmp.shape) == 3:
        img = img[:, :, :3]

    img_prediction = get_prediction(img, s, model, means, stds, img_patch_size, img_border, img_total_size,
                                    num_channels, eval_batch_size, num_labels)

    oimg = make_img_overlay(img, img_prediction)

    return oimg
