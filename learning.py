"""

New version of the code for clear understanding of the method with better modularization.

"""

import tensorflow as tf
import numpy as np
import sys
import progressbar

import prediction_helpers as pred_help
import image_helpers as img_help
import global_vars

# Initialisation of some flags for tensor flow
# (In this case we declare the directory to store nets as we go)

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/mnist',
    """Directory where to write event logs and checkpoint.""")

FLAGS = tf.app.flags.FLAGS

SEED = global_vars.SEED

# Function that computes whatever score you want given a scoring function
def compute_score(scoring_f, data, labels, session, model):

    pred = pred_help.get_prediction_from_patches(
        data, session, model, global_vars.EVAL_BATCH_SIZE,
        global_vars.IMG_TOTAL_SIZE, global_vars.NUM_CHANNELS,
        global_vars.NUM_LABELS)

    score = scoring_f(pred, labels)
    return score

def seperate_set(data, labels):
    indices = np.random.permutation(data.shape[0])

    train_len = int(global_vars.VALIDATION_TRAIN_PERC * len(indices))
    valid_len = int(global_vars.VALIDATION_VAL_PERC * len(indices))

    train_idx = indices[:train_len]
    valid_idx = indices[train_len:train_len+valid_len]
    test_idx = indices[train_len+valid_len:]

    train_data = data[train_idx]
    train_labels = labels[train_idx]
    train_data, means, stds = img_help.standardize(train_data)

    valid_data = data[valid_idx]
    valid_labels = labels[valid_idx]
    valid_data, _, _ = img_help.standardize(valid_data, means=means, stds=stds)

    test_data = data[test_idx]
    test_labels = labels[test_idx]
    test_data, _, _ = img_help.standardize(test_data, means=means, stds=stds)

    return ((train_data, train_labels), 
        (valid_data, valid_labels), 
        (test_data, test_labels))

def preparing_data(sat_images, label_images):

    # Adding rotated images
    if global_vars.ROTATE_IMAGES:
        perm = np.random.choice(sat_images.shape[0], sat_images.shape[0])
        imgs = []
        labels = []

        for i in perm[0:global_vars.ROTATED_IMG]:
            angle = np.random.rand() * 360
            rot_img = img_help.rotate_image(sat_images[i], angle)
            rot_label = img_help.rotate_image(label_images[i], angle)
            imgs.append(rot_img)
            labels.append(rot_label)

        sat_images = np.append(sat_images, imgs, axis=0)
        label_images = np.append(label_images, labels, axis=0)

    print("we have a total of", len(sat_images), "images for training.")

    # Extrcting patches from images
    data = img_help.extract_data(
        sat_images, global_vars.IMG_PATCH_SIZE, global_vars.IMG_BORDER)

    labels = img_help.extract_labels(label_images, global_vars.IMG_PATCH_SIZE)

    return data, labels

def main(argv=None):
    # setup seeds
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    # output and input files
    output_run_log = 'runs.txt'
    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'

    # arrays to store the different scores
    f1_validation_per_epoch = []
    f1_training_per_epoch = []
    loss_per_recording_step = []
    
    # File regex to load the images for training and testing
    FILE_REGEX = "satImage_%.3d"
    
    # Getting training images
    sat_images, label_images = img_help.read_images(
        train_data_filename, train_labels_filename, 
        global_vars.TRAINING_SIZE, FILE_REGEX)

    # Getting the data on ehich we are going to train
    data, labels = preparing_data(sat_images, label_images)

    # Seperating our data in three distinct sets (taining, validation, testing)
    (train_set, valid_set, test_set) = seperate_set(data, labels)

    # Balancing data
    train_set = img_help.balance_data(train_set[0], train_set[1])

    print("We will train on", len(train_set[0]), "patches of size", 
        str(global_vars.IMG_TOTAL_SIZE)+ "x" + str(global_vars.IMG_TOTAL_SIZE))

    num_epochs = global_vars.NUM_EPOCHS

    # Initialization of placeholders for data and labels
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(
            global_vars.BATCH_SIZE, global_vars.IMG_TOTAL_SIZE, 
            global_vars.IMG_TOTAL_SIZE, global_vars.NUM_CHANNELS))

    train_label_node = tf.placeholder(
        tf.float32,
        shape=(global_vars.BATCH_SIZE, global_vars.NUM_LABELS))

    eval_data_node = tf.placeholder(
        tf.float32,
        shape=(global_vars.EVAL_BATCH_SIZE, global_vars.IMG_TOTAL_SIZE, 
            global_vars.IMG_TOTAL_SIZE, global_vars.NUM_CHANNELS))

    eval_label_node = tf.placeholder(
        tf.float32,
        shape=(global_vars.EVAL_BATCH_SIZE, global_vars.NUM_LABELS))

    # Define the parameters of the convolutional layers
    conv_params, last_depth = params_conv_layers(
        global_vars.CONV_ARCH, global_vars.CONV_DEPTH, global_vars.NUM_CHANNELS)

    pool_fact = 2 ** len(global_vars.CONV_ARCH)

    if global_vars.IMG_TOTAL_SIZE % pool_fact != 0:
        raise "not dividable by pool fact " + str(global_vars.IMG_TOTAL_SIZE) + " / " + str(pool_fact)

    size = int(global_vars.IMG_TOTAL_SIZE / pool_fact * global_vars.IMG_TOTAL_SIZE / pool_fact * last_depth)

    fc_params = params_fc_layers(global_vars.FC_ARCH, global_vars.FC_DEPTH, size, global_vars.NUM_LABELS)

    # Definition of the complete cnn model.
    def model(data, train=False):

        # convolution layers
        conv_end = init_conv_layers(global_vars.CONV_ARCH, conv_params, data)

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        conv_end_shape = conv_end.get_shape().as_list()

        reshape = tf.reshape(
            conv_end,
            [conv_end_shape[0], conv_end_shape[1] * conv_end_shape[2] * conv_end_shape[3]])

        print(reshape.get_shape())

        out = init_fc_layers(global_vars.FC_ARCH, fc_params, reshape, train)

        return out

    logits = model(train_data_node, True)

    # Computes the probability error for each prediction
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits, train_label_node))
    tf.summary.scalar('loss', loss)

    # L2 regularization for the fully connected parameters.
    regularizers = tf.nn.l2_loss(fc_params[0][0]) + tf.nn.l2_loss(fc_params[0][1])
    for params in fc_params[1:]:
        regularizers += tf.nn.l2_loss(params[0])
        regularizers += tf.nn.l2_loss(params[1])
    
    # Add the regularization term to the loss.
    loss += 5e-4 * (regularizers)

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)

    # Use adam optimizer as it optimises automatically the learning rate.
    adam_opt = tf.train.AdamOptimizer(global_vars.LEARNING_RATE)
    optimizer = adam_opt.minimize(loss, global_step=batch)

    # Predictions for the minibatch, validation set and test set.
    train_prediction = tf.nn.softmax(logits)
    # Compute predictions for validation and test
    correct_prediction_train = tf.equal(
        tf.argmax(train_prediction,1), tf.argmax(train_label_node,1))
    # Accuracy for training
    accuracy_train = tf.reduce_mean(tf.cast(correct_prediction_train, tf.float32))

    # Validation / Testing set predictions
    predictions = tf.nn.softmax(model(eval_data_node))
    # Compute predictions for validation and test
    correct_prediction = tf.equal(tf.argmax(predictions,1), tf.argmax(eval_label_node,1))
    # Accuracy for test as a sum, as we will have to do a mean by patch
    accuracy_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    truePos = tf.reduce_sum(
        tf.cast(
            tf.equal(tf.argmax(tf.boolean_mask(predictions, correct_prediction), 1), 1),
            tf.float32))

    falsePos = tf.reduce_sum(
        tf.cast(
            tf.equal( tf.argmax(tf.boolean_mask(predictions, tf.logical_not(correct_prediction)), 1), 1),
            tf.float32))
            
    falseNeg = tf.reduce_sum(
        tf.cast(
            tf.equal(
                tf.argmax(tf.boolean_mask(predictions, tf.logical_not(correct_prediction)), 1), 0),
                tf.float32))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # initialise all varibales operation
    init = tf.global_variables_initializer()

    s = tf.Session()

    if global_vars.RESTORE_MODEL:
        # Restore variables from disk.
        saver.restore(s, FLAGS.train_dir + "/model.ckpt")
        print("Model restored.")
    else:
        # run initialisation of variables
        s.run(init)

        print('Initialized!')
        # Loop through training steps.
        print('Total number of iterations : ' + str(int(num_epochs * len(train_set[0]) / global_vars.BATCH_SIZE)))

        train_size = len(train_set[0])

        i = 0
        epoch_bar = progressbar.ProgressBar(max_value=num_epochs)

        try:
            batch_size = global_vars.BATCH_SIZE
            for epoch in range(num_epochs):
                print("training for epoch", epoch)
                epoch_bar.update(epoch)
                perm_idx = np.random.permutation(train_size)

                batch_bar = progressbar.ProgressBar(max_value=int(train_size / global_vars.BATCH_SIZE))
                for step in range(int(train_size / global_vars.BATCH_SIZE)):
                    batch_idx = perm_idx[step * batch_size : (step+1) * batch_size]

                    # Compute the offset of the current minibatch in the data.
                    # Note that we could use better randomization across epochs.
                    batch_data = train_set[0][batch_idx]
                    batch_labels = train_set[1][batch_idx]

                    # This dictionary maps the batch data (as a np array) to the
                    # node in the graph is should be fed to.
                    feed_dict = {train_data_node: batch_data,
                                 train_label_node: batch_labels}
                    
                    if step % global_vars.RECORDING_STEP == 0:
                        _, train_acc, l = s.run(
                            [optimizer, accuracy_train, loss], feed_dict=feed_dict)
                        
                        print("Recording step, might take a long time")
                        
                        acc = batch_sum(s, accuracy_sum, valid_set, global_vars.EVAL_BATCH_SIZE, eval_data_node, eval_label_node)

                        valid_acc = acc / (int(len(valid_set[0]) / global_vars.EVAL_BATCH_SIZE) * global_vars.EVAL_BATCH_SIZE)

                        print('%.2f' % (float(step) * global_vars.BATCH_SIZE / train_size) + '% of Epoch ' + str(epoch + 1))
                        print("loss :",l)
                        print("training set accuracy :", train_acc)
                        print("validation set accuracy :", valid_acc)

                        # TODO: do a logging function
                        loss_per_recording_step.append(l)

                        sys.stdout.flush()
                    else:
                        batch_bar.update(step)
                        s.run(optimizer, feed_dict=feed_dict)
                    
                # What do here ? nothing normally as done at beginning of each epoch
        except KeyboardInterrupt:
            print("Interrupted at epoch ", epoch + 1)
            pass

        print("Scoring on validation set")

        acc = batch_sum(s, accuracy_sum, valid_set, global_vars.EVAL_BATCH_SIZE, eval_data_node, eval_label_node)
        accuracy = acc / (int(len(valid_set[0]) / global_vars.EVAL_BATCH_SIZE) * global_vars.EVAL_BATCH_SIZE)

        print("Accuracy rating is :", accuracy)

        print("Scoring on testing set")

        acc = batch_sum(s, accuracy_sum, test_set, global_vars.EVAL_BATCH_SIZE, eval_data_node, eval_label_node)
        accuracy = acc / (int(len(test_set[0]) / global_vars.EVAL_BATCH_SIZE) * global_vars.EVAL_BATCH_SIZE)

        print("Accuracy rating is :", accuracy)



                    
def batch_sum(s, func_sum, data_set, eval_batch_size, eval_data_node, eval_label_node):
    # TODO: put this in a function
    # Evaluating accuracy for EVAL_BATCH_SIZE parts of the validation set
    set_len = len(data_set[0])
    batch_nbr = int(set_len / eval_batch_size) + 1
    batch_idxs = np.array_split(range(set_len), batch_nbr)
    
    b_update = 0
    score_bar = progressbar.ProgressBar(max_value=len(batch_idxs))

    acc = 0
    
    for batch_idx in batch_idxs:
        score_bar.update(b_update)
        b_update += 1

        if len(batch_idx) < eval_batch_size:
            batch_idx = range(set_len)[-eval_batch_size:]

        feed_dict = {eval_data_node: data_set[0][batch_idx],
            eval_label_node: data_set[1][batch_idx]}

        acc += s.run(func_sum, feed_dict=feed_dict)

    return acc


def init_fc_layers(fc_arch, fc_params, prev_layer, dropout):
    # convolution layers

    for i in range(fc_arch):
        print(i)
        prev_layer = tf.nn.relu(tf.matmul(prev_layer, fc_params[i][0]) + fc_params[i][1])
        if dropout:
            prev_layer = tf.nn.dropout(prev_layer, 0.5, seed=SEED)

    fc_end = tf.matmul(prev_layer, fc_params[-1][0]) + fc_params[-1][1]

    return fc_end  

def init_conv_layers(conv_arch, conv_params, prev_layer):
    # convolution layers

    for i, n_conv in enumerate(conv_arch):

        for layer in range(n_conv):
            # 2D convolution
            conv = tf.nn.conv2d(
                prev_layer,
                conv_params[i][layer][0],
                strides=[1, 1, 1, 1],
                padding='SAME')

            # Bias and rectified linear non-linearity.
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv_params[i][layer][1]))
            prev_layer = relu

        prev_layer = tf.nn.max_pool(
            prev_layer,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME')

    return prev_layer

def params_conv_layers(conv_arch, conv_depth, channels):
    # init of conv parameters
    conv_params = [None] * len(conv_arch)

    # init origin depth
    prev_depth = channels
    for i, n_conv in enumerate(conv_arch):
        conv_params[i] = [None] * n_conv
        new_depth = conv_depth[i]
        for layer in range(n_conv):
            conv_weights = tf.Variable(
                tf.truncated_normal(
                    [3, 3, prev_depth, new_depth],  # 3x3 filter, depth augmenting by few steps.
                    stddev=0.1,
                    seed=SEED))
            conv_biases = tf.Variable(tf.zeros([new_depth]))
            conv_params[i][layer] = (conv_weights, conv_biases)
            prev_depth = new_depth
    
    print(new_depth)

    return conv_params, new_depth

def params_fc_layers(fc_arch, fc_depth, depth_conv, last_depth):

    fc_param = []

    print(depth_conv)

    prev_depth = depth_conv
    for i in range(fc_arch):
        weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal(
                [prev_depth, fc_depth[i]],
                stddev=0.1,
                seed=SEED))
        biases = tf.Variable(tf.constant(0.1, shape=[fc_depth[i]]))
        fc_param.append((weights, biases))
        prev_depth = fc_depth[i]

    last_weights = tf.Variable(
        tf.truncated_normal(
            [prev_depth, last_depth],
            stddev=0.1,
            seed=SEED))
    last_biases = tf.Variable(tf.constant(0.1, shape=[last_depth]))
    fc_param.append((last_weights, last_biases))

    return fc_param


if __name__ == '__main__':
    tf.app.run()
    

    