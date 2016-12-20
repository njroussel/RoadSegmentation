import os

import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from scipy import ndimage

from global_vars import *


def balance_data(data, labels):
    c0 = 0
    c1 = 0
    for i in range(len(labels)):
        if labels[i][0] == 1:
            c0 += 1
        else:
            c1 += 1

    # Make populations even.
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    data = data[new_indices]
    labels = labels[new_indices, :]
    return data, labels


def rotate_image(image, angle):
    rotated_image = ndimage.rotate(image, angle, mode='reflect', order=0, reshape=False)
    return rotated_image


def read_binary_images(image_filename, num_images, file_regex):
    images = []
    for i in range(1, num_images + 1):
        imageid = file_regex % i
        filename = image_filename + imageid + ".png"
        if os.path.isfile(filename):
            print('Loading ' + filename)
            img = mpimg.imread(filename)
            images.append(img)
        else:
            print('File ' + filename + ' does not exist')

    return np.array(images)


def read_3channel_images(image_filename, num_images, file_regex):
    images = []

    for i in range(1, num_images + 1):
        imageid = file_regex % i
        filename = image_filename + imageid + ".png"

        if os.path.isfile(filename):
            print('Loading ' + filename)
            img = mpimg.imread(filename)
            tmp = np.array(img)
            if len(tmp.shape) == 3:
                img = img[:, :, :3]

            images.append(img)
        else:
            print('File ' + filename + ' does not exist')

    return np.array(images)


def read_images(train_filename, label_filename, num_images, file_regex):
    return (
        read_3channel_images(train_filename, num_images, file_regex),
        read_binary_images(label_filename, num_images, file_regex))


def quantize_binary_images(images, quantization_patch_size, output_patch_size):
    quantized_images = []
    for image in images:
        single_pixel_image = extract_labels([image], quantization_patch_size).reshape(
            (-1, int(image.shape[0] / quantization_patch_size), 2))[:, :, 0].T
        output_patch_image = ndimage.zoom(single_pixel_image, output_patch_size, order=0)
        quantized_images.append(output_patch_image)
    output = np.array(quantized_images)
    output = output.reshape(output.shape[0], output.shape[1], output.shape[2], 1)
    return output


def standardize(images, means=None, stds=None):
    """ Standardize a group of patches.
        Returns std_patches, means, stds.
            @param images : Patches to standardize.
    """
    if images.shape[3] == 1:
        layer = images[:, :, :, 0]
        if means is None:
            mean = np.mean(layer)
        else:
            mean = means[0]
        std_layer = layer - mean
        if stds is None:
            std = np.std(layer)
        else:
            std = stds[0]
        if std > 0:
            std_layer /= std

        std_data = std_layer.reshape(std_layer.shape[0], std_layer.shape[1], std_layer.shape[2], 1)
        return std_data, [mean], [std]
    else:
        r_layer = images[:, :, :, 0]
        g_layer = images[:, :, :, 1]
        b_layer = images[:, :, :, 2]

        if means is None:
            r_mean = np.mean(r_layer)
            g_mean = np.mean(g_layer)
            b_mean = np.mean(b_layer)
        else:
            r_mean = means[0]
            g_mean = means[1]
            b_mean = means[2]

        std_r_layer = r_layer - r_mean
        std_g_layer = g_layer - g_mean
        std_b_layer = b_layer - b_mean

        if stds is None:
            r_std = np.std(r_layer)
            g_std = np.std(g_layer)
            b_std = np.std(b_layer)
        else:
            r_std = stds[0]
            g_std = stds[1]
            b_std = stds[2]

        if r_std > 0:
            std_r_layer /= r_std
        if g_std > 0:
            std_g_layer /= g_std
        if b_std > 0:
            std_b_layer /= b_std

        std_data = np.stack((std_r_layer, std_g_layer, std_b_layer), axis=3)
        return std_data, [r_mean, g_mean, b_mean], [r_std, g_std, b_std]


# Extract patches from a given image
def img_crop(im, w, h, border=0):
    """ Crop an image into 'patches'.
        @param im : The image to crop (array).
        @param w : width of a patch.
        @param h : height of a patch.
    """
    list_patches = []
    img_width = im.shape[0]
    img_height = im.shape[1]
    if border != 0:
        im = np.array([np.pad(im[:, :, i], ((border, border), (border, border)), 'symmetric').T
                       for i in range(im.shape[2])
                       ]).T
    for i in range(0, img_height, h):
        for j in range(0, img_width, w):
            im_patch = im[j:j + w + 2 * border, i:i + h + 2 * border]
            list_patches.append(im_patch)
    return list_patches


def extract_data(images, patch_size, border):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
        @param images : the images.
        @param border : the border.
    """
    num_images = len(images)
    img_patches = [img_crop(images[i], patch_size, patch_size, border) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    return np.asarray(data)


def value_to_class(v):
    """ Assign a label to a patch given its color mean.
        @param v : mean label of the image/patch.
    """
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df < foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]


def extract_labels(images, patch_size):
    """ Extract the labels into a 1-hot matrix [image index, label index].
        @param images : the images.
    """
    data = extract_data(images, patch_size, 0)
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)


def error_rate(predictions, labels):
    """ Compute error rate, that is the percentage of wrong predictions.
        @param predictions : Array of predictions.
        @param labels : Array of expected labels.
    """
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0])


def write_predictions_to_file(predictions, labels, filename):
    # TODO: doc to be confirmed.
    """ Writes the predictions to a file.
        @param predictions : The computed predictions.
        @param labels : The labels.
        @param filename : File in which all of this will be written.
    """
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
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
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    print(str(max_labels) + ' ' + str(max_predictions))


def label_to_img(imgwidth, imgheight, w, h, labels, thresh=0.5):
    """ Create a binary image from labels.
        @param imgwidth : image width.
        @param imgheight : image height.
        @param w : width of a patch.
        @param h : height of a patch.
        @param labels : labels of the patches.
    """
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx][0] > thresh:
                l = 1
            else:
                l = 0
            array_labels[j:j + w, i:i + h] = l
            idx += 1
    return array_labels


def img_float_to_uint8(img):
    """ Convert a float image to a uint8 one.
        @param img : The image to be converted.
    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
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
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def make_img_overlay(img, predicted_img):
    """ Draw red patches on the satellite image.
        @param img : The original image.
        @param predicted_img : The label.
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img
