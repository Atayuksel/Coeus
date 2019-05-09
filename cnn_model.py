import tensorflow as tf
import functools
import math

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class CNNModel:
    def __init__(self, data, target, conv_filters_size, conv_filters_out, max_pool_filter_sizes, class_weights,
                 conv_stride, hidden_unit_size, embedding_size, vocabulary_size, dropout, learning_rate, seq_lens):

        self.data = data
        self.target = target
        self.seq_lens = seq_lens
        self.class_weights = class_weights

        self.dropout = dropout
        self.learning_rate = learning_rate
        self.vocab_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_unit_size = hidden_unit_size
        self.conv_stride = conv_stride
        self.mp_filter_size = max_pool_filter_sizes

        self.max_seq_len = self.data.get_shape()[1].value
        self.batch_size = self.data.get_shape()[0].value
        self.target_size = self.target.get_shape()[1].value

        # fc_input_size_x = math.ceil(((((self.max_seq_len - conv_filters_size[0]) / max_pool_filter_sizes[0])
        #                               - conv_filters_size[2]) / max_pool_filter_sizes[2]))
        #
        # fc_input_size_y = math.ceil(((((self.embedding_size - conv_filters_size[1]) / max_pool_filter_sizes[1])
        #                               - conv_filters_size[3]) / max_pool_filter_sizes[3]))

        fc_input_size_x = math.ceil(((self.max_seq_len / max_pool_filter_sizes[0]) / max_pool_filter_sizes[2]) / max_pool_filter_sizes[4])
        fc_input_size_y = math.ceil(((self.embedding_size / max_pool_filter_sizes[1]) / max_pool_filter_sizes[3]) / max_pool_filter_sizes[5])

        fc_input_size = fc_input_size_x * fc_input_size_y * conv_filters_out[2]

        # network variables (weights, biases, embeddings)
        self.weights = {
            # 1st convolution layer weights
            'wc1': tf.Variable(tf.random_normal([conv_filters_size[0], conv_filters_size[1],
                                                 1, conv_filters_out[0]])),
            # 2nd convolution layer weights
            'wc2': tf.Variable(tf.random_normal([conv_filters_size[2], conv_filters_size[3],
                                                 conv_filters_out[0], conv_filters_out[1]])),
            # 3nd convolution layer weights
            'wc3': tf.Variable(tf.random_normal([conv_filters_size[4], conv_filters_size[5],
                                                 conv_filters_out[1], conv_filters_out[2]])),
            # perceptron layer weights
            'wd1': tf.Variable(tf.random_normal([fc_input_size, self.hidden_unit_size])),
            # output layer weights
            'out': tf.Variable(tf.random_normal([self.hidden_unit_size, self.target_size]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([conv_filters_out[0]])),
            'bc2': tf.Variable(tf.random_normal([conv_filters_out[1]])),
            'bc3': tf.Variable(tf.random_normal([conv_filters_out[2]])),
            'bd1': tf.Variable(tf.random_normal([hidden_unit_size])),
            'out': tf.Variable(tf.random_normal([self.target_size]))
        }

        self.embedding_v = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embedding_size]),
                                       trainable=True,
                                       name="word_embedding_variable",
                                       dtype=tf.float32)

        self.prediction
        self.optimize
        self.error

    def conv2d(self, data, conv_weight, b):
        data = tf.nn.conv2d(data, conv_weight, strides=[1, self.conv_stride, self.conv_stride, 1], padding='SAME')
        data = tf.nn.bias_add(data, b)
        return tf.nn.relu(data)

    def maxpool2d(self, data, k1, k2):
        return tf.nn.max_pool(data, ksize=[1, k1, k2, 1], strides=[1, k1, k2, 1], padding='SAME')

    @lazy_property
    def prediction(self):
        # get data and reshape it wrt network
        embedding_data = tf.nn.embedding_lookup(self.embedding_v, self.data)
        data = tf.reshape(embedding_data, [self.batch_size, self.max_seq_len, self.embedding_size, 1])

        # 1st convolution layer
        conv1 = self.conv2d(data, self.weights['wc1'], self.biases['bc1'])
        # 1st max pooling layer
        conv1 = self.maxpool2d(conv1, self.mp_filter_size[0], self.mp_filter_size[1])

        # 2nd convolution layer
        conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        # 2nd max pooling layer
        conv2 = self.maxpool2d(conv2, self.mp_filter_size[2], self.mp_filter_size[3])

        # 3nd convolution layer
        conv3 = self.conv2d(conv2, self.weights['wc3'], self.biases['bc3'])
        # 3nd max pooling layer
        conv3 = self.maxpool2d(conv3, self.mp_filter_size[4], self.mp_filter_size[5])

        # fully connected layer
        # reshape conv2 data for fully connected layer
        fc1 = tf.reshape(conv3, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)

        # apply dropout
        fc1 = tf.nn.dropout(fc1, self.dropout)

        # output, class predictions
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
        return out

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
        return tf.cast(mistakes, tf.float32)
