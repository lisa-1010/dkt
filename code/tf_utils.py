
import os, datetime
import tensorflow as tf
import tflearn
from tflearn.config import _EPSILON, _FLOATX

from utils import *

def build_char_level_ast_dynamic_rnn_encoder(maxlen=100, vocab_size=50, embedding_size=64, hidden_dim=128):
    """
    Takes in a sequence of characters from a program represented as a abstract syntax tree (AST),
    and uses an LSTM RNN to produce an encoding. This function simply specifies the computation graph,
    it does not add any regression / optimization layers.

    Expects each input sample to represent a sequence of characters, where each character is represented by a number.

    the output of the net is a single encoding for the entire ast. The length of the encoding vector is specified
    through the parameter hidden_dim.

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
    net = tflearn.regression(y_pred, optimizer='adam', loss='categorical_crossentropy')
    return net


def sequential_categorical_crossentropy(y_pred, y_true, mask=None):
    """ Categorical Crossentropy.

    Computes cross entropy between y_pred (logits) and y_true (labels).

    Measures the probability error in discrete classification tasks in which
    the classes are mutually exclusive (each entry is in exactly one class).
    For example, each CIFAR-10 image is labeled with one and only one label:
    an image can be a dog or a truck, but not both.

    `y_pred` and `y_true` must have the same shape `[batch_size, num_timesteps, num_classes]`
    and the same dtype (either `float32` or `float64`). It is also required
    that `y_true` (labels) are binary arrays (For example, class 2 out of a
    total of 5 different classes, will be define as [0., 1., 0., 0., 0.])

    Arguments:
        y_pred: `2D-Tensor`, flattened from 3D-Tensor. Predicted values.
        y_true: `2D-Tensor`, flattened from 3D-Tensor. Targets (labels), a probability distribution.
        mask: `2D-Tensor`, flattened from 3D-Tensor. masks out the padded timesteps. (since sequences are not the equal length)

    """
    with tf.name_scope("Sequential_Crossentropy"):
        y_pred /= tf.reduce_sum(y_pred,
                                reduction_indices=len(y_pred.get_shape())-1,
                                keep_dims=True)
        # manual computation of crossentropy
        y_pred = tf.clip_by_value(y_pred, tf.cast(_EPSILON, dtype=_FLOATX),
                                  tf.cast(1.-_EPSILON, dtype=_FLOATX))
        if mask:
            seq_cross_entropy = - tf.reduce_sum(y_true * mask * tf.log(y_pred),
                                                reduction_indices=len(y_pred.get_shape()) - 1)
        else:
            seq_cross_entropy = - tf.reduce_sum(y_true * tf.log(y_pred),
                                   reduction_indices=len(y_pred.get_shape())-1)
        return tf.reduce_mean(seq_cross_entropy)

# def build_lstm_net_predict_sequence(n_timesteps=10, n_inputdim=50, n_hidden=128, n_classes=2, mask=None):
#     """
#     builds a tensorflow lstm with prediction at every timestep.
#     Input: A time series of embeddings. Each embedding models an AST, the time series contains the embeddings
#     of the ASTs in a trajectory.
#     Output: A time series of real-valued numbers, predicting the number of steps left to completion of the problem.
#     """
#     net = tflearn.input_data([None, n_timesteps, n_inputdim])
#
#     net = tflearn.lstm(net, n_hidden, return_seq=True)
#     net = tflearn.dropout(net, 0.5)
#     # y_pred = tflearn.time_distributed(net, tflearn.fully_connected, [2, 'softmax', True, 'truncated_normal', 'zeros', None, 0.001, True, True, True, "FullyConnected"]) # 2 for binary
#     # y_pred = tflearn.time_distributed(net, tflearn.fully_connected,
#     #                                   [2, 'softmax'], scope="FullyConnected")
#     y_pred = tflearn.lstm(net, n_classes, return_seq=True)
#     # outgoing tensor of shape [None, max_seq_len, 2]
#     y_pred = tf.reshape(y_pred, [-1, n_timesteps * n_classes], name='Reshape')
#     # mask y_pred to exclude certain timesteps
#     net = tflearn.regression(y_pred, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
#     return net





def build_lstm_net_predict_sequence(n_timesteps=10, n_inputdim=50, n_hidden=128, n_classes=2, mask=None):
    """
    builds a tensorflow lstm with prediction at every timestep.
    Input: A time series of embeddings. Each embedding models an AST, the time series contains the embeddings
    of the ASTs in a trajectory.
    Output: A time series of real-valued numbers, predicting the number of steps left to completion of the problem.
    """

    X = tf.placeholder(shape=(None, n_timesteps, n_inputdim), dtype=tf.float32, name="X")
    mask = tf.placeholder(shape=(None, n_timesteps), dtype=tf.int32, name="mask")
    Y = tf.placeholder(shape=n_classes, dtype=tf.float32, name="Y")

    tf.add_to_collection(tf.GraphKeys.INPUTS, X)
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + "X", X)

    tf.add_to_collection(tf.GraphKeys.INPUTS, mask)
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + "mask", mask)

    if Y not in tf.get_collection(tf.GraphKeys.TARGETS):
        tf.add_to_collection(tf.GraphKeys.TARGETS, Y)

    net = tflearn.lstm(X, n_hidden, return_seq=True)
    net = tflearn.dropout(net, 0.5)
    # y_pred = tflearn.time_distributed(net, tflearn.fully_connected, [2, 'softmax', True, 'truncated_normal', 'zeros', None, 0.001, True, True, True, "FullyConnected"]) # 2 for binary
    # y_pred = tflearn.time_distributed(net, tflearn.fully_connected,
    #                                   [2, 'softmax'], scope="FullyConnected")
    net = tflearn.lstm(net, n_classes, return_seq=True)
    # outgoing tensor of shape [None, max_seq_len, 2]
    net = tf.reshape(net, [-1, n_timesteps * n_classes], name='Reshape')
    # mask y_pred to exclude certain timesteps

    return net


def build_lstm_net(n_timesteps=10, n_inputdim=50, n_hidden=128,  n_classes=2):
    """
    lstm with prediction only at last timestep.
    Input: A time series of embeddings. Each embedding models an AST, the time series contains the embeddings
    of the ASTs in a trajectory.
    Output: A time series of real-valued numbers, predicting the number of steps left to completion of the problem.
    """
    net = tflearn.input_data([None, n_timesteps, n_inputdim])
    net = tflearn.lstm(net, n_hidden, return_seq=False, dynamic=True)
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, n_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')
    return net




def build_two_layer_lstm_net(n_timesteps=10, n_inputdim=50, n_hidden=128,  n_classes=2):
    """
    lstm with prediction only at last timestep.
    Input: A time series of embeddings. Each embedding models an AST, the time series contains the embeddings
    of the ASTs in a trajectory.
    Output: A time series of real-valued numbers, predicting the number of steps left to completion of the problem.
    """
    net = tflearn.input_data([None, n_timesteps, n_inputdim], name='input')
    net = tflearn.lstm(net, n_hidden, return_seq=True, dynamic=True, name='lstm_1')
    net = tflearn.dropout(net, 0.5)
    net = tflearn.lstm(net, n_hidden/2, return_seq=False, name='lstm_2')
    net = tflearn.fully_connected(net, n_classes, activation='softmax', name='fc')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')
    return net


if __name__ == '__main__':
    print ("Testing the module tf_utils.py. ")
    #
    # model = load_model(model_id='lstm_1', load_checkpoint=True, is_training=True)