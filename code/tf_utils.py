
import os, datetime
import tensorflow as tf
import tflearn

from utils import *

def build_char_level_ast_dynamic_rnn_encoder(maxlen=100, vocab_size=50, embedding_size=64, hidden_dim=128):
    """
    Takes in a sequence of characters from a program represented as a abstract syntax tree (AST),
    and uses an LSTM RNN to produce an encoding. This function simply specifies the computation graph,
    it does not add any regression / optimization layers.

    Expects each input sample to represent a sequence of characters, where each character is represented by a number.

    the output of the net is a single encoding for the entire ast. The length of the encoding vector is specified
    through the parameter hidden_dim.
    Parameters
    ----------
    maxlen
    vocab_size
    embedding_size
    hidden_dim

    Returns
    -------

    """
    net = tflearn.layers.input_data([None, maxlen])
    net = tflearn.embedding(net, input_dim=vocab_size, output_dim=embedding_size)
    net = tflearn.lstm(net, hidden_dim, dropout=0.8, dynamic=True, return_seq=False)
    return net

def build_rnn_from_ast_encoder_to_student_success(net):
    # incoming tensor of shape [None, max_seq_len, hidden_dim]
    y_pred = tflearn.time_distributed(net, tflearn.fully_connected, [2,'softmax'])
    # outgoing tensor of shape [None, max_seq_len, 2]
    y_pred = tflearn.flatten(y_pred) # make sure to also flatten y_true
    net = tflearn.regression(y_pred, optimizer='adam', loss='binary_crossentropy')
    return net



def build_lstm_net(n_timesteps=100, n_inputdim=50, n_hidden=128):
    """
    builds a tensorflow lstm with two layers.
    Input: A time series of embeddings. Each embedding models an AST, the time series contains the embeddings
    of the ASTs in a trajectory.
    Output: A time series of real-valued numbers, predicting the number of steps left to completion of the problem.
    Parameters
    ----------
    n_timesteps
    n_inputdim
    n_hidden
    n_outputdim

    Returns
    -------

    """
    net = tflearn.input_data([None, n_timesteps, n_inputdim])
    net = tflearn.lstm(net, n_hidden, return_seq=True)
    net = tflearn.dropout(net, 0.5)
    y_pred = tflearn.time_distributed(net, tflearn.fully_connected, [2, 'softmax']) # 2 for binary
    # outgoing tensor of shape [None, max_seq_len, 2]
    y_pred = tflearn.flatten(y_pred)  # make sure to also flatten y_true
    net = tflearn.regression(y_pred, optimizer='adam', loss='binary_crossentropy')
    return net



def load_model(model_id, load_checkpoint=False, is_training=False):
    # should be used for all models
    print ('Loading model...')

    if model_id == 'lstm_1':
        net = build_lstm_net()

    tensorboard_dir = '../tensorboard_logs/' + model_id + '/'
    checkpoint_path = '../checkpoints/' + model_id + '/'

    print (tensorboard_dir)
    print (checkpoint_path)

    check_if_path_exists_or_create(tensorboard_dir)
    check_if_path_exists_or_create(checkpoint_path)

    if is_training:
        model = tflearn.DNN(net, tensorboard_verbose=2, tensorboard_dir=tensorboard_dir, \
                            checkpoint_path=checkpoint_path, max_checkpoints=3)
    else:
        model = tflearn.DNN(net)

    if load_checkpoint:
        checkpoint = tf.train.latest_checkpoint(checkpoint_path)  # can be none of no checkpoint exists
        if checkpoint and os.path.isfile(checkpoint):
            model.load(checkpoint)
            print ('Checkpoint loaded.')
        else:
            print ('No checkpoint found. ')

    print ('Model loaded.')
    return model


# def train_model(model_id, dataset_name, use_checkpoint=True):
#
#     x, y = load_data(dataset=dataset_name) #TODO: write load_data function
#
#     print ('x shape: {}'.format(x.shape))
#     print ('y shape: {}'.format(y.shape))
#
#     model = load_model(model_id, load_checkpoint=use_checkpoint, training_mode=True)
#
#     date_time_string = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
#     run_id = "{}_{}".format(model_id, date_time_string)
#
#     model.fit(x, y, n_epoch=32, validation_set=0.1, show_metric=True, snapshot_step=100, run_id=run_id)


if __name__ == '__main__':
    print ("Testing the module tf_utils.py. ")

    model = load_model(model_id='lstm_1', load_checkpoint=True, is_training=True)