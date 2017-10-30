import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys


def dot_semantic_nn(context, utterance, tng_mode):
    """
    Creates a two channel network that encodes two different inputs and calculates
    a dot product to measure how close they are.

    :param context: Embedding of summed word embeddings for the context (1 x emb_dim)
    :param utterance: Embedding of summed word embeddings for the utterance (1 x emb_dim)
    :param tng_mode: tf ModeKeys key
    :return:
    """

    # disable dropout during inference
    # only time it's used is during training
    keep_prob = 0.5
    if tng_mode == ModeKeys.TRAIN:
        keep_prob = 0.5

    # -----------------------------
    # CONTEXT CHANNEL
    # this channel encodes the context
    # this is a (1 x emb_size) vector
    context_channel = _network_channel(network_name='context_channel', net_input=context, keep_prob=keep_prob)

    # -----------------------------
    # UTTERANCE CHANNEL
    # this channel encodes the utterance
    # this is a (1 x emb_size) vector
    utterance_channel = _network_channel(network_name='utterance_channel', net_input=utterance, keep_prob=keep_prob)

    # -----------------------------
    # LOSS
    # negative log probability while using K-1 examples in the batch
    # as negative samples
    mean_loss = _negative_log_probability_loss(context_channel, utterance_channel)
    K = tf.matmul(context_channel, utterance_channel, transpose_b=True)

    # return the loss and the encoding from each channel
    return mean_loss, context_channel, utterance_channel, K


def _negative_log_probability_loss(context_channel, utterance_channel):
    """
    This implements the negative log probability using negative sampling
    where K-1 items in the batch are treated as negative samples where
    i != j.

    The overall loss formula is:
    $$L(x,y,\theta) = -\frac{1}{K}\sum_{i=1}^{K}{[ f(x_i, y_i) - log \sum_{j=1}^{K}{e^{f(x_i,y_i)}}]}$$

    :param context_channel:
    :param utterance_channel:
    :return:
    """
    # calculate dot product between each pair of inputs and responses
    # (bs x bs)
    K = tf.matmul(context_channel, utterance_channel, transpose_b=True)

    # get the diagonals which are the S(x_i, y_i)
    # this represents the similarity score between each input x_i and output y_i
    # out = (bs x 1)
    S = tf.diag_part(K)
    S = tf.reshape(S, [-1, 1])

    # calculate the log sum(e^x_i, y_j)
    # here every row has only the negative examples
    # in = (bs x bs).  out = (bs x 1)
    K = tf.reduce_logsumexp(K, axis=1, keep_dims=True)

    # compute the mean loss between each x,y pair
    # and the log sum of each other (K-1) x,y pair
    return -tf.reduce_mean(S - K)


def _network_channel(network_name, net_input, keep_prob):
    """
    Generates an n layer Dense network that encodes the inputs
    into a k dimensional space
    :param network_name:
    :param net_input:
    :param keep_prob:
    :return:
    """
    with tf.variable_scope(network_name) as scope:
        predict_opt_name = '{}_branch_predict'.format(network_name)

        # use 3 dense layers for this network branch
        with tf.variable_scope('dense_branch') as d_scope:
            dense_0 = tf.layers.dense(net_input, units=300, activation=tf.nn.tanh)
            dense_0 = tf.layers.batch_normalization(dense_0)
            dense_0 = tf.layers.dropout(inputs=dense_0, rate=keep_prob)

            dense_1 = tf.layers.dense(dense_0, units=300, activation=tf.nn.tanh)
            dense_1 = tf.layers.batch_normalization(dense_1)
            dense_1 = tf.layers.dropout(inputs=dense_1, rate=keep_prob)

            dense_2 = tf.layers.dense(dense_1, units=500, activation=tf.nn.tanh, name=predict_opt_name)
            tf.add_to_collection('{}_embed_opt'.format(network_name), dense_2)

        return dense_2
