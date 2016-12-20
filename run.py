##Hello darkness

import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

import image_helpers as img_help
import learner as learn

SEED = 0xDEADBEEF

import prediction_helpers as pred_help
import global_vars
import global_vars_pp


# Initialisation of some flags for tensor flow
# (In this case we declare the directory to store nets as we go))

def main(argv=None):
    # setup seeds
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    tf.app.flags.DEFINE_string(
        'train_dir', '/tmp/model',
        """Directory where to write event logs and checkpoint.""")

    flags = tf.app.flags.FLAGS

    # Create save directory if needed
    if not os.path.exists(flags.train_dir):
        os.makedirs(flags.train_dir)

    save_model_folder = "model_save/"
    save_model_file_name = "last_model"

    # Getting arguments for model folder
    if len(sys.argv) == 2:
        save_model_file_name = sys.argv[1]
    if len(sys.argv) == 3:
        save_model_folder = sys.argv[1]
        save_model_file_name = sys.argv[2]

    save_model_file_name += ".ckpt"

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

    s, model, means, stds, max_thresh = learn.learn(sat_images, label_images, global_vars, SEED, flags,
                                                    save_model_folder + save_model_file_name)

    if global_vars.TRAIN_PREDICTIONS:
        ## Run on test set.
        print("\n******************************************************************************")
        print('Running on train set\n')
        file_regex = 'satImage_%.3d'
        test_data_filename = './training/images/'
        test_dir = 'predictions_training/'
        if not os.path.isdir(test_dir):
            os.mkdir(test_dir)
        for i in range(1, global_vars.TRAINING_SIZE + 1):
            print('train prediction {}'.format(i))
            p_img = pred_help.get_prediction_image(test_data_filename, i, s, model, file_regex, means, stds,
                                                   global_vars, max_thresh)
            Image.fromarray(p_img).save(test_dir + "prediction_" + str(i) + ".png")
            oimg = pred_help.get_prediction_with_overlay(test_data_filename, i, s, model, file_regex, means, stds,
                                                         global_vars, max_thresh)
            oimg.save(test_dir + "overlay_" + str(i) + ".png")

    if global_vars.TEST_PREDICTIONS:
        ## Run on test set.
        print("\n******************************************************************************")
        print('Running on test set\n')
        file_regex = 'test_%d'
        TEST_SIZE = 50
        test_data_filename = './test_set_images/'
        test_dir = 'test_predictions/'
        if not os.path.isdir(test_dir):
            os.mkdir(test_dir)
        for i in range(1, TEST_SIZE + 1):
            print('test prediction {}'.format(i))
            p_img = pred_help.get_prediction_image(test_data_filename, i, s, model, file_regex, means, stds,
                                                   global_vars, max_thresh)
            Image.fromarray(p_img).save(test_dir + "prediction_" + str(i) + ".png")
            oimg = pred_help.get_prediction_with_overlay(test_data_filename, i, s, model, file_regex, means, stds,
                                                         global_vars, max_thresh)
            oimg.save(test_dir + "overlay_" + str(i) + ".png")

    ###############################################
    ###############################################
    ###############################################
    ###############################################
    ###############################################

    tf.app.flags.DEFINE_string('train_dir', '/tmp/model', """Directory where to write event logs and checkpoint.""")

    flags = tf.app.flags.FLAGS

    # Create save directory if needed
    if not os.path.exists(flags.train_dir):
        os.makedirs(flags.train_dir)

    save_model_folder = "model_save/"
    save_model_file_name = "pp_model"

    ## TODO : USE INPUT CORRECTLY
    # Getting arguments for model folder
    if len(sys.argv) == 2:
        save_model_file_name = sys.argv[1]
    if len(sys.argv) == 3:
        save_model_folder = sys.argv[1]
        save_model_file_name = sys.argv[2]

    save_model_file_name += ".ckpt"

    # input files
    train_data_filename = './predictions_training/'
    train_labels_filename = './training/groundtruth/'

    # Getting training images
    print("\nLoading images :")
    print("******************************************************************************")
    file_regex = "prediction_%d"
    prediction_images = img_help.read_binary_images(train_data_filename, global_vars_pp.TRAINING_SIZE, file_regex)
    file_regex = "satImage_%.3d"
    label_images = img_help.read_binary_images(train_labels_filename, global_vars_pp.TRAINING_SIZE, file_regex)

    prediction_images = img_help.quantize_binary_images(prediction_images, global_vars.IMG_PATCH_SIZE,
                                                        global_vars_pp.IMG_PATCH_SIZE)
    label_images = img_help.quantize_binary_images(label_images, global_vars.IMG_PATCH_SIZE,
                                                   global_vars_pp.IMG_PATCH_SIZE)

    s, model, means, stds, max_thresh = learn.learn(prediction_images, label_images, global_vars_pp, SEED, flags,
                                                    save_model_folder + save_model_file_name)

    if global_vars_pp.TRAIN_PREDICTIONS:
        ## Run on test set.
        print("\n******************************************************************************")
        print('Running on train set\n')
        file_regex = 'prediction_%d'
        test_data_filename = './test_predictions/'
        test_dir = 'predictions_training_post/'
        if not os.path.isdir(test_dir):
            os.mkdir(test_dir)
        for i in range(1, global_vars_pp.TRAINING_SIZE + 1):
            print('train prediction {}'.format(i))
            p_img = pred_help.get_prediction_image(test_data_filename, i, s, model, file_regex, means, stds,
                                                   global_vars_pp, max_thresh)
            p_img = \
                img_help.quantize_binary_images([p_img], global_vars_pp.IMG_PATCH_SIZE, global_vars.IMG_PATCH_SIZE)[
                    0]
            p_img = p_img.reshape(p_img.shape[0], p_img.shape[1])
            p_img = img_help.img_float_to_uint8(p_img)

            Image.fromarray(p_img).save(test_dir + "prediction_" + str(i) + ".png")

    if global_vars_pp.TEST_PREDICTIONS:
        ## Run on test set.
        print("\n******************************************************************************")
        print('Running on test set\n')
        file_regex = 'prediction_%d'
        TEST_SIZE = 50
        test_data_filename = './test_predictions/'
        test_dir = 'test_predictions_post/'
        if not os.path.isdir(test_dir):
            os.mkdir(test_dir)
        for i in range(1, TEST_SIZE + 1):
            print('test prediction {}'.format(i))
            p_img = pred_help.get_prediction_image(test_data_filename, i, s, model, file_regex, means, stds,
                                                   global_vars_pp, max_thresh)
            p_img = \
                img_help.quantize_binary_images([p_img], global_vars_pp.IMG_PATCH_SIZE, global_vars.IMG_PATCH_SIZE)[
                    0]

            p_img = p_img.reshape(p_img.shape[0], p_img.shape[1])
            p_img = img_help.img_float_to_uint8(p_img)
            Image.fromarray(p_img).save(test_dir + "prediction_" + str(i) + ".png")


if __name__ == '__main__':
    tf.app.run()
