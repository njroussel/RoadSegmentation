#!/usr/bin/env python3

import re

import sys
import matplotlib.image as mpimg
import numpy as np

foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch


# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(img, tmp):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", img).group(0))
    im = mpimg.imread(img)
    for j in range(0, im.shape[1], tmp):
        for i in range(0, im.shape[0], tmp):
            patch = im[i:i + tmp, j:j + tmp]
            label = patch_to_label(patch)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, PATCH_SIZE))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        dir_name = sys.argv[1]
    else:
        dir_name = 'test_predictions/'
    PATCH_SIZE = 16
    submission_filename = 'dummy_submission.csv'
    image_filenames = []
    for i in range(1, 51):
        image_filename = dir_name + 'prediction_' + '%.d' % i + '.png'
        print (image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(submission_filename, *image_filenames)
