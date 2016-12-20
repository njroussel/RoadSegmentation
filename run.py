##Hello darkness

import os
import sys

import numpy as np
import tensorflow as tf

import global_vars
import global_vars_pp
import image_helpers as img_help
import learner as learn
import prediction_helpers as pred_help

SEED = 0xDEADBEEF

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/model', """Directory where to write event logs and checkpoint.""")

FLAGS = tf.app.flags.FLAGS

# Create save directory if needed
if not os.path.exists(FLAGS.train_dir):
    os.makedirs(FLAGS.train_dir)


# Initialisation of some flags for tensor flow
# (In this case we declare the directory to store nets as we go))
def main(argv=None):
    # setup seeds
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    save_model_folder = "model_save/"
    save_model_file_name = "last_model"

    # Getting arguments for model folder
    if len(sys.argv) == 2:
        save_model_file_name = sys.argv[1]

    save_model_file_name += ".ckpt"

    # output and input files
    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'
    save_model_file_path = save_model_folder + save_model_file_name

    # File regex to load the images for training and testing
    file_regex = "satImage_%.3d"

    # Getting training images
    print("\nLoading images :")
    print("******************************************************************************")
    sat_images, label_images = img_help.read_images(
        train_data_filename, train_labels_filename, global_vars.TRAINING_SIZE, file_regex)

    s, model, means, stds, max_thresh = learn.learn(
        sat_images, label_images, global_vars, SEED, FLAGS, save_model_file_path)

    if global_vars.POST_PROCESS:
        # Run on train set.
        print("\n******************************************************************************")
        print('Running on train set\n')

        file_regex = 'satImage_%.3d'
        input_dir = './training/images/'
        output_dir = 'predictions_training/'

        pred_help.get_prediction_images(file_regex, input_dir, output_dir, global_vars.TRAINING_SIZE, s, model,
                                        means, stds, global_vars, max_thresh, False)

    if global_vars.TEST_PREDICTIONS:
        # Run on test set.
        print("\n******************************************************************************")
        print('Running on test set\n')

        file_regex = 'test_%d'
        test_size = 50
        input_dir = './test_set_images/'
        output_dir = 'test_predictions/'

        pred_help.get_prediction_images(file_regex, input_dir, output_dir, test_size, s, model,
                                        means, stds, global_vars, max_thresh, True)

    ###############################################
    ###############################################
    ###############################################
    ###############################################
    ###############################################

    if global_vars.POST_PROCESS:
        tf.app.flags.DEFINE_string(
            'train_dir', '/tmp/model', """Directory where to write event logs and checkpoint.""")

        flags = tf.app.flags.FLAGS

        # Create save directory if needed
        if not os.path.exists(flags.train_dir):
            os.makedirs(flags.train_dir)

        save_model_folder = "save_model_pp/"

        # input files
        train_data_filename = './predictions_training/'
        train_labels_filename = './training/groundtruth/'
        save_model_file_path = save_model_folder + save_model_file_name

        # Getting training images
        print("\nLoading images :")
        print("******************************************************************************")

        file_regex = "prediction_%d"
        prediction_images = img_help.read_binary_images(
            train_data_filename, global_vars_pp.TRAINING_SIZE, file_regex)

        file_regex = "satImage_%.3d"
        label_images = img_help.read_binary_images(
            train_labels_filename, global_vars_pp.TRAINING_SIZE, file_regex)

        prediction_images = img_help.quantize_binary_images(
            prediction_images, global_vars_pp.IMG_PATCH_SIZE, global_vars_pp.IMG_PATCH_SIZE)

        label_images = img_help.quantize_binary_images(
            label_images, global_vars_pp.IMG_PATCH_SIZE, global_vars_pp.IMG_PATCH_SIZE)

        s, model, means, stds, max_thresh = learn.learn(
            prediction_images, label_images, global_vars_pp, SEED, FLAGS, save_model_file_path)

        if global_vars_pp.TEST_PREDICTIONS:
            # Run on test set.
            print("\n******************************************************************************")
            print('Running on test set\n')
            file_regex = 'prediction_%d'
            test_size = 50
            input_dir = './test_predictions/'
            output_dir = 'test_predictions_post/'

            pred_help.get_prediction_images(file_regex, input_dir, output_dir, test_size, s, model,
                                            means, stds, global_vars, max_thresh, False)


if __name__ == '__main__':
    tf.app.run()
