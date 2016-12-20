##Hello darkness

import numpy as np
import tensorflow as tf
import os
from PIL import Image

import image_helpers as img_help

import learner as learn

SEED = 0xDEADBEEF

import prediction_helpers as pred_help
import global_vars
import global_vars_pp

# Initialisation of some flags for tensor flow
# (In this case we declare the directory to store nets as we go)

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/model',
    """Directory where to write event logs and checkpoint.""")

FLAGS = tf.app.flags.FLAGS

# Create save directory if needed
if not os.path.exists(FLAGS.train_dir):
    os.makedirs(FLAGS.train_dir)


def main(argv=None):
    # setup seeds
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    # output and input files
    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'

    # File regex to load the images for training and testing
    file_regex = "satImage_%.3d"

    # Getting training images
    print("\nLoading images :")
    print("******************************************************************************")
    sat_images, label_images = img_help.read_images(
        train_data_filename, train_labels_filename,
        global_vars.TRAINING_SIZE, file_regex)

    s, model, means, stds, max_thresh = learn.learn(sat_images, label_images, SEED, FLAGS)

    if global_vars.TRAIN_PREDICTIONS:
        ## Run on test set.
        print("\n******************************************************************************")
        print('Running on train set\n')
        FILE_REGEX = 'satImage_%.3d'
        test_data_filename = './training/images/'
        test_dir = 'predictions_training/'
        if not os.path.isdir(test_dir):
            os.mkdir(test_dir)
        for i in range(1, global_vars.TRAINING_SIZE + 1):
            print('train prediction {}'.format(i))
            pimg = pred_help.get_prediction_image(test_data_filename, i, s, model, FILE_REGEX, means, stds,
                                                  global_vars, max_thresh)
            Image.fromarray(pimg).save(test_dir + "prediction_" + str(i) + ".png")
            oimg = pred_help.get_prediction_with_overlay(test_data_filename, i, s, model, FILE_REGEX, means, stds,
                                                         global_vars, max_thresh)
            oimg.save(test_dir + "overlay_" + str(i) + ".png")

    if global_vars.TEST_PREDICTIONS:
        ## Run on test set.
        print("\n******************************************************************************")
        print('Running on test set\n')
        FILE_REGEX = 'test_%d'
        TEST_SIZE = 50
        test_data_filename = './test_set_images/'
        test_dir = 'test_predictions/'
        if not os.path.isdir(test_dir):
            os.mkdir(test_dir)
        for i in range(1, TEST_SIZE + 1):
            print('test prediction {}'.format(i))
            pimg = pred_help.get_prediction_image(test_data_filename, i, s, model, FILE_REGEX, means, stds,

                                                  global_vars, max_thresh)
            Image.fromarray(pimg).save(test_dir + "prediction_" + str(i) + ".png")
            oimg = pred_help.get_prediction_with_overlay(test_data_filename, i, s, model, FILE_REGEX, means, stds,
                                                         global_vars, max_thresh)
            oimg.save(test_dir + "overlay_" + str(i) + ".png")


if __name__ == '__main__':
    tf.app.run()
