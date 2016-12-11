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
def get_prediction_with_groundtruth(filename, image_idx):
    global FILE_REGEX
    imageid = FILE_REGEX % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)
    tmp = numpy.array(img)
    if len(tmp.shape) == 3:
        img = img[:, :, :3]

    img_prediction = get_prediction(img)
    return img_float_to_uint8(img_prediction)


# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(filename, image_idx):
    global FILE_REGEX
    imageid = FILE_REGEX % image_idx
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)
    tmp = numpy.array(img)
    if len(tmp.shape) == 3:
        img = img[:, :, :3]

    img_prediction = get_prediction(img)
    oimg = make_img_overlay(img, img_prediction)

    return oimg
