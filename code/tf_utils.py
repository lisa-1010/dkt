

import tensorflow as tf
import tflearn
import datetime


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



def build_lstm_net(n_timesteps, n_inputdim, n_hidden=128, n_outputdim):
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
    # maxlen = 100 # max sequence length
    # net = tflearn.input_data([None, n_timesteps, n_inputdim])
    net = tflearn.layers.input_data([None, n_timesteps, n_inputdim])
    net = tflearn.layers.lstm(net, n_hidden, return_seq=True)
    net = tflearn.layers.dropout(net, 0.5)
    y_pred = tflearn.time_distributed(net, tflearn.fully_connected, [2, 'softmax'])
    # outgoing tensor of shape [None, max_seq_len, 2]
    y_pred = tflearn.flatten(y_pred)  # make sure to also flatten y_true
    net = tflearn.regression(y_pred, optimizer='adam', loss='binary_crossentropy')
    return net



def load_model(model_id, load_ckpt=False):
    net = build_lstm_net()

    model = None
    return model


def train_model(model_id, dataset_name, use_checkpoint=True):

    x, y = load_data(dataset=dataset_name)

    print ('x shape: {}'.format(x.shape))
    print ('y shape: {}'.format(y.shape))

    # check_model_exists_or_create_new(model_id, n_timesteps, n_inputdim, n_hidden, n_classes, architecture)

    model = load_model(model_id, load_checkpoint=use_checkpoint, training_mode=True)
    print ("Model loaded.")

    date_time_string = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    run_id = "{}_{}".format(model_id, date_time_string)
    print ("Run id: {}".format(run_id))

    model.fit(x, y, n_epoch=32, validation_set=0.1, show_metric=True, snapshot_step=100, run_id=run_id)
