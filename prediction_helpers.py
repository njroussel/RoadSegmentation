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


# Get a concatenation of the prediction and groundtruth for given input file
def get_prediction_with_groundtruth(filename, image_idx, s, model, file_regex, means, stds):
    imageid = file_regex % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)
    tmp = numpy.array(img)
    if len(tmp.shape) == 3:
        img = img[:, :, :3]

    img_prediction = get_prediction(img, s, model, means, stds)
    return img_float_to_uint8(img_prediction)

# Helper function to run multiple baches instead of one big dataset (crashes most GPU)
def eval_in_batches(data, sess, eval_prediction, data_node):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]

    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)

    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)

    for begin in range(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={data_node: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={data_node: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

# Get prediction for given input image
def get_prediction(img, s, model, means, stds):
    data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE, border=IMG_BORDER))
    data, _, _ = standardize(data, means, stds)
    data_node = tf.placeholder(
        tf.float32, 
        shape=(EVAL_BATCH_SIZE, IMG_TOTAL_SIZE, IMG_TOTAL_SIZE, NUM_CHANNELS))

    output = tf.nn.softmax(model(data_node))
    output_prediction = eval_in_batches(data, s, output, data_node)

    img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)
    return img_prediction


def get_prediction_from_patches(patches, s, model):
    data = numpy.asarray(patches)
    
    data_node = tf.placeholder(
        tf.float32, 
        shape=(EVAL_BATCH_SIZE, IMG_TOTAL_SIZE, IMG_TOTAL_SIZE, NUM_CHANNELS))

    output = tf.nn.softmax(model(data_node))
    output_prediction = eval_in_batches(data, s, output, data_node)
    return output_prediction
    # img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)
    # return img_prediction


# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(filename, image_idx, s, model, file_regex, means, stds):
    imageid = file_regex % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)
    tmp = numpy.array(img)
    if len(tmp.shape) == 3:
        img = img[:, :, :3]

    img_prediction = get_prediction(img, s, model, means, stds)
    oimg = make_img_overlay(img, img_prediction)

    return oimg
