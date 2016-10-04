

import tensorflow as tf
import tflearn

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

def build_rnn_from_ast_encoder_to_success(net):
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
    # net = tflearn.layers.lstm(net, n_hidden, return_seq=True)
    # net = tflearn.layers.dropout(net, 0.5)
    # output from prev layer is a
    net = tflearn.layers.fully_connected(net, 10, activation='softmax')
    net = tflearn.layers.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')



def load_model(model_id, load_ckpt=False):
    net = build_lstm_net()

    pass