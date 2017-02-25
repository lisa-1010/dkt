import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import tensorflow as tf
import numpy as np
import os, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tf_utils import *
from utils import *
from models_dict import *



def load_model(model_id, load_checkpoint=False, is_training=False, get_hidden_rep=False, timesteps=None):
    # should be used for all models
    print ('Loading model...')

    model_dict = load_model_dict(model_id)
    n_timesteps, n_inputdim, n_hidden, n_classes, architecture = model_dict["n_timesteps"], model_dict["n_inputdim"], \
                                                                  model_dict["n_hidden"], model_dict["n_classes"], \
                                                                 model_dict["architecture"]

    if timesteps:
        # if provided as an argument, overwrite n_timesteps from the model
        n_timesteps = timesteps

    if architecture == 'lstm_predict_binary_sequence':
        net = build_lstm_net_predict_sequence(n_timesteps=n_timesteps, n_inputdim=n_inputdim, n_hidden=n_hidden, n_classes=n_classes, mask=None)

    elif architecture == 'lstm_predict_binary':
        net = build_lstm_net(n_timesteps=n_timesteps, n_inputdim=n_inputdim, n_hidden=n_hidden, n_classes=n_classes)

    elif architecture == 'two_layer_lstm_predict_binary':
        net = build_two_layer_lstm_net(n_timesteps=n_timesteps, n_inputdim=n_inputdim, n_hidden=n_hidden,
                                       n_classes=n_classes, get_hidden_rep=get_hidden_rep)

    tensorboard_dir = '../tensorboard_logs/' + model_id + '/'
    checkpoint_path = '../checkpoints/' + model_id + '/'
    best_checkpoint_path = '../best_checkpoints/' + model_id + '/'

    print (tensorboard_dir)
    print (checkpoint_path)
    print (best_checkpoint_path)

    check_if_path_exists_or_create(tensorboard_dir)
    check_if_path_exists_or_create(checkpoint_path)
    check_if_path_exists_or_create(best_checkpoint_path)

    if is_training:
        model = tflearn.DNN(net, tensorboard_verbose=2, tensorboard_dir=tensorboard_dir, \
                            checkpoint_path=checkpoint_path, best_checkpoint_path=best_checkpoint_path, max_checkpoints=3)
    else:
        model = tflearn.DNN(net)

    if load_checkpoint:
        checkpoint = tf.train.latest_checkpoint(checkpoint_path)  # can be none of no checkpoint exists
        print checkpoint
        if checkpoint and os.path.isfile(checkpoint):
            model.load(checkpoint, weights_only=True, verbose=True)
            print ('Checkpoint loaded.')
        else:
            print ('No checkpoint found. ')

    print ('Model loaded.')
    return model


def get_results_saved_model(model_id, minlen):
    graph_to_use = tf.Graph()
    with graph_to_use.as_default():
        saved_model = load_model(model_id, load_checkpoint=True, is_training=False)
        x, y = load_data_will_student_solve_next_problem(hoc_num=18, minlen=minlen)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        pred_train =np.argmax(saved_model.predict(x_train), axis=1)
        pred_test = np.argmax(saved_model.predict(x_test), axis=1)
        train_acc = accuracy_score(pred_train, np.argmax(y_train, axis=1))
        test_acc = accuracy_score(pred_test,  np.argmax(y_test, axis=1))
        print ("Train acc: {}\t Test acc: {}".format(train_acc, test_acc))
        return train_acc, test_acc


def get_results_saved_model_traj_len(hoc_num, model_id, traj_len):
    graph_to_use = tf.Graph()
    with graph_to_use.as_default():
        saved_model = load_model(model_id, load_checkpoint=True, is_training=False, timesteps=traj_len)
        x, y, student_ids = load_data_will_student_solve_next_problem_traj_len(hoc_num, only_traj_len=traj_len, y_is_seq=False)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        pred_train =np.argmax(saved_model.predict(x_train), axis=1)
        pred_test = np.argmax(saved_model.predict(x_test), axis=1)
        train_acc = accuracy_score(pred_train, np.argmax(y_train, axis=1))
        test_acc = accuracy_score(pred_test,  np.argmax(y_test, axis=1))
        print ("Train acc: {}\t Test acc: {}".format(train_acc, test_acc))
        return train_acc, test_acc, student_ids


