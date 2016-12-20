NUM_CHANNELS = 1  # Binary images
NUM_LABELS = 2
TRAINING_SIZE = 100
SEED = 43212  # Set to None for random seed.
EVAL_BATCH_SIZE = 128  # 64
BATCH_SIZE = 16
NUM_EPOCHS = 30  # Will later be determined by validation.
ROTATION_AMOUNT = 3
ROTATE_IMAGES = True
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
TRAIN_PREDICTIONS = False  # If True, restore existing model instead of training a new one
TEST_PREDICTIONS = True
ENABLE_RECORDING = False
RECORDING_STEP = 1000
LEARNING_RATE = 0.001

# Convolution network architecture
CONV_ARCH = [1]

# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 2  # MUST BE EVEN

# Border for enhanced context.
IMG_BORDER = 10  # MUST BE EVEN (could eventually be even ....)

IMG_TOTAL_SIZE = IMG_PATCH_SIZE + 2 * IMG_BORDER

# Validation parameters:
VALIDATION_TRAIN_PERC = 0.6
VALIDATION_VAL_PERC = 0.3
VALIDATION_TEST_PERC = 0.1

# Hyperparameters validation
COMPUTE_VALIDATION_F1_SCORE_FOR_EACH_EPOCH = True