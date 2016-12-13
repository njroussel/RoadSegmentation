NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 10
SEED = 43212  # Set to None for random seed.
EVAL_BATCH_SIZE = 64  # 64
BATCH_SIZE = 16  # 64
NUM_EPOCHS = 2
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
TRAIN_PREDICTIONS = False  # If True, restore existing model instead of training a new one
TEST_PREDICTIONS = True
ENABLE_RECORDING = False
RECORDING_STEP = 1000
LEARNING_RATE = 0.001

# Convolution network architecture
CONV_ARCH = [2, 2, 3]

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

# Border for enhanced context.
IMG_BORDER = 4

IMG_TOTAL_SIZE = IMG_PATCH_SIZE + 2 * IMG_BORDER

# Validation parameters:
VALIDATION_TRAIN_PERC = 0.7
VALIDATION_VAL_PERC = 0.2
VALIDATION_TEST_PERC = 0.1
