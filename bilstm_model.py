import tensorflow as tf
from tensorflow.contrib import rnn
import functools


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class BiLSTMModel:
    def __init__(self, data, target, seq_lens, class_weights, num_hidden, learning_rate, embedding_size, vocab_size):
        self.data = data
        self.target = target
        self.seq_lens = seq_lens
        self.class_weights = class_weights
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.vocabulary_size = vocab_size

        seq_len = self.data.get_shape()[1]
        target_size = self.target.get_shape()[1]

        self.weights = {
            'out': tf.Variable(tf.random_normal([2 * self.num_hidden, target_size.value]))}
        self.biases = {
            'out': tf.Variable(tf.random_normal([target_size.value]))}
        self.embedding_v = tf.Variable(tf.constant(0.0, shape=[self.vocabulary_size, self.embedding_size]),
                                       trainable=True,
                                       name="word_embedding_variable",
                                       dtype=tf.float32)

        self.prediction
        self.optimize
        self.error

    def bilstm_layer(self, data, keep_prob):
        max_seq_len = data.get_shape()[1]
        x = tf.unstack(data, max_seq_len, 1)
        lstm_fw_cell = rnn.LSTMBlockCell(num_units=self.num_hidden)
        lstm_fw_cell_dropout = rnn.DropoutWrapper(cell=lstm_fw_cell,
                                                  input_keep_prob=keep_prob,
                                                  output_keep_prob=keep_prob,
                                                  state_keep_prob=keep_prob)
        lstm_bw_cell = rnn.LSTMBlockCell(num_units=self.num_hidden)
        lstm_bw_cell_dropout = rnn.DropoutWrapper(cell=lstm_bw_cell,
                                                  input_keep_prob=keep_prob,
                                                  output_keep_prob=keep_prob,
                                                  state_keep_prob=keep_prob)
        rnn_output, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell_dropout, lstm_bw_cell_dropout, x,
                                                        sequence_length=self.seq_lens,
                                                        dtype=tf.float32)
        return rnn_output

    def lstm_max_pooling(self, rnn_output):
        lstm_output = tf.stack(values=rnn_output, axis=1)
        result = []
        for i in range(lstm_output.shape[0].value):
            sentence_length = self.seq_lens[i]
            output = lstm_output[i]
            output = output[:sentence_length]
            output = tf.reduce_max(input_tensor=output,
                                   axis=0,
                                   keepdims=False)
            result.append(output)
        output = tf.stack(values=result, axis=0)
        return output

    @lazy_property
    def prediction(self):
        embedding_data = tf.nn.embedding_lookup(self.embedding_v, self.data)
        rnn_output = self.bilstm_layer(embedding_data, 0.5)
        max_pooling_output = self.lstm_max_pooling(rnn_output)
        logits = tf.matmul(max_pooling_output, self.weights['out']) + self.biases['out']
        return logits

    @lazy_property
    def optimize(self):
        weights = tf.reduce_sum(self.class_weights * self.target, axis=1)
        unweighted_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction,
                                                                                       labels=self.target))
        weighted_loss = unweighted_loss_op * weights
        loss = tf.reduce_mean(weighted_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                           beta1=0.9,
                                           beta2=0.999)
        return optimizer.minimize(loss)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        # return tf.reduce_mean(tf.cast(mistakes, tf.float32))
        return tf.cast(mistakes, tf.float32)
