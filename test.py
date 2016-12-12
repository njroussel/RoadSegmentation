from image_helpers import *

test = [[[[2, 1, 3], [3, 1, 3]], [[4, 1, 3], [5, 1, 3]]], [[[6, 1, 3], [7, 1, 3]], [[8, 1, 3], [9, 1, 3]]]]
test = numpy.array(test)

data, means, stds = standardize(test)
print(means)