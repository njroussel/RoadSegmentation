NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 20
SEED = 66478  # Set to None for random seed.
EVAL_BATCH_SIZE = 64  # 64
BATCH_SIZE = 64  # 64
NUM_EPOCHS = 20
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
TRAIN_PREDICTIONS = False  # If True, restore existing model instead of training a new one
TEST_PREDICTIONS = True
ENABLE_RECORDING = False
RECORDING_STEP = 1000
LEARNING_RATE = 0.00001

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

# Border for enhanced context.
IMG_BORDER = 16

IMG_TOTAL_SIZE = IMG_PATCH_SIZE + 2 * IMG_BORDER

# Validation parameters:
VALIDATION_TRAIN_PERC = 0.6
VALIDATION_VAL_PERC = 0.3
VALIDATION_TEST_PERC = 0.1
