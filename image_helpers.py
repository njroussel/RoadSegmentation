import os

import matplotlib.image as mpimg
import numpy
from PIL import Image

from global_vars import *


def img_crop(im, w, h):
    """ Crop an image into 'patches'.
        @param im : The image to crop (array).
        @param w : width of a patch.
        @param h : height of a patch.
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)
    return list_patches


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
        @param filename : path to the images.
        @param num_images : number of images in the folder.
    """
    imgs = []
    for i in range(1, num_images + 1):
        global FILE_REGEX
        imageid = FILE_REGEX % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            tmp = numpy.array(img)
            if len(tmp.shape) == 3:
                img = img[:, :, :3]
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    return numpy.asarray(data)


def value_to_class(v):
    """ Assign a label to a patch given its color mean.
        @param v : mean label of the image/patch.
    """
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df < foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]


def extract_labels(filename, num_images):
    """ Extract the labels into a 1-hot matrix [image index, label index].
        @param filename : folder containing images.
        @param num_images : number of images in that folder.
    """
    gt_imgs = []
    for i in range(1, num_images + 1):
        global FILE_REGEX
        imageid = FILE_REGEX % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


def error_rate(predictions, labels):
    """ Compute error rate, that is the percentage of wrong predictions.
        @param predictions : Array of predictions.
        @param labels : Array of expected labels.
    """
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])


def write_predictions_to_file(predictions, labels, filename):
    # TODO: doc to be confirmed.
    """ Writes the predictions to a file.
        @param predictions : The computed predictions.
        @param labels : The labels.
        @param filename : File in which all of this will be written.
    """
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()


def print_predictions(predictions, labels):
    # TODO: doc to be confirmed.
    """ Print the predictions in stdout.
        @param predictions : The computed predictions.
        @param labels : The labels.
    """
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print(str(max_labels) + ' ' + str(max_predictions))


def label_to_img(imgwidth, imgheight, w, h, labels):
    """ Create a binary image from labels.
        @param imgwidth : image width.
        @param imgheight : image height.
        @param w : width of a patch.
        @param h : height of a patch.
        @param labels : labels of the patches.
    """
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx][0] > 0.5:
                l = 1
            else:
                l = 0
            array_labels[j:j + w, i:i + h] = l
            idx = idx + 1
    return array_labels


def img_float_to_uint8(img):
    """ Convert a float image to a uint8 one.
        @param img : The image to be converted.
    """
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg


def concatenate_images(img, gt_img):
    """ Produce results images side by side [stallite|groundtruth].
        @param img : The satellite image.
        @param gt_img : The corresponding groundtruth image.
    """
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def make_img_overlay(img, predicted_img):
    """ Draw red patches on the satellite image.
        @param img : The original image.
        @param predicted_img : The label.
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:, :, 0] = predicted_img * PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img
