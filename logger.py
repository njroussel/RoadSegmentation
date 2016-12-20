from global_vars import *
import time

params_file_name = 'runs.txt'

# What we will log:
logs = {
    "Accuracy_training": [],
    "Accuracy_validation": [],
    "Accuracy_test": 0,

    "F1-score_validation": [],
    "F1-score_threshs_validation": [],

    "F1-score_test": 0,

    "Epoch_stop": 0,

    "Training_size": TRAINING_SIZE,
    "Seed": SEED,
    "Batch_size": BATCH_SIZE,
    "Num_epoch": NUM_EPOCHS,
    "Theta_thresh": THETA_THESH, 
    "Rotate_images": ROTATE_IMAGES,
    "Nbr_rotated_img": ROTATED_IMG,
    "Conv_arch": CONV_ARCH,
    "Conv_depth": CONV_DEPTH, 
    "Fc_arch": FC_ARCH, 
    "Fc_depth": FC_DEPTH, 
    "Img_border": IMG_BORDER, 
    "Validation_train_perc": VALIDATION_TRAIN_PERC, 
    "Validation_val_perc": VALIDATION_VAL_PERC, 
    "Validation_test_perc": VALIDATION_TEST_PERC, 
}

def append_log(who, what):
    logs[who].append(what)

def set_log(who, what):
    logs[who] = what

def save_log():
    param_file = open(params_file_name, 'a')
    param_file.write("Date={}:\n".format(time.strftime("%c")))
    for key, value in logs.items():
        param_file.write(key + "=" + str(value) + "\n") 
    param_file.write("################################################################################\n\n")
    param_file.close()