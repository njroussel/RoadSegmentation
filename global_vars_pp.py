NUM_CHANNELS = 1  # Binary images
NUM_LABELS = 2
TRAINING_SIZE = 100
SEED = 43212  # Set to None for random seed.
EVAL_BATCH_SIZE = 128  # 64
BATCH_SIZE = 128
NUM_EPOCHS = 10  # Will later be determined by validation.
ROTATED_IMG = 300
ROTATE_IMAGES = True
RESTORE_MODEL = True  # If True, restore existing model instead of training a new one
TEST_PREDICTIONS = True
RECORDING_STEP = 1000
LEARNING_RATE = 0.001

FILTER_SIZE = 3
INPUT_PATCH_SIZE = 16  # MUST BE THE SAME AS PATCH SIZE IN GLOBAL_VARS

OPTI_F1 = True

# Convolution network architecture
CONV_ARCH = [2]
KEEP_DROPOUT = 0.8
CONV_DEPTH = [32, 64, 128, 256]

FC_ARCH = 2
FC_DEPTH = [1024, 1024]
# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 2  # MUST BE EVEN

# Border for enhanced context.
IMG_BORDER = 30  # MUST BE EVEN (could eventually be even ....)

IMG_TOTAL_SIZE = IMG_PATCH_SIZE + 2 * IMG_BORDER

THETA_THRESH = 0.01

# Validation parameters:
VALIDATION_TRAIN_PERC = 0.6
VALIDATION_VAL_PERC = 0.2
VALIDATION_TEST_PERC = 0.2

# Hyperparameters validation
COMPUTE_VALIDATION_F1_SCORE_FOR_EACH_EPOCH = True
