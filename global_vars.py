NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 100
SEED = 43212  # Set to None for random seed.
EVAL_BATCH_SIZE = 64  # 64
BATCH_SIZE = 64
NUM_EPOCHS = 100 # Will later be determined by validation.
ROTATE_IMAGES = False
ROTATED_IMG = 300
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
POST_PROCESS = False
TEST_PREDICTIONS = True
RECORDING_STEP = 1000
LEARNING_RATE = 0.001
OPTI_F1 = False
FILTER_SIZE = 3

KEEP_DROPOUT = 0.8

# Convolution network architecture
CONV_ARCH = [1, 1] # The best architecture so far on validation.
# [2, 4, 4, 6] Best on test set.
# We will keep both, but mostly use [2, 2, 4] because of our limited
# computation power.
CONV_DEPTH = [32, 64, 128, 256]

FC_ARCH = 1
FC_DEPTH = [512]

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

# Border for enhanced context.
IMG_BORDER = 16  # Will be set after archi is found

IMG_TOTAL_SIZE = IMG_PATCH_SIZE + 2 * IMG_BORDER

THETA_THRESH = 0.01

# Validation parameters:
VALIDATION_TRAIN_PERC = 0.6
VALIDATION_VAL_PERC = 0.2
VALIDATION_TEST_PERC = 0.2

# Hyperparameters validation
COMPUTE_VALIDATION_F1_SCORE_FOR_EACH_EPOCH = True
