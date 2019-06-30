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

class KimCNN:
    def __init__(self, data, target, data_pos_tags, distance_protein, distance_chemical, embedding_placeholder,
                 hidden_unit_size, filter_size, embedding_size, vocabulary_size, learning_rate, position_embedding_size,
                 pos_tag_embedding_size,
                 max_position_distance):

        self.data = data
        self.target = target
        self.distance_to_protein = distance_protein
        self.distance_to_chemical = distance_chemical
        self.pos_tags = data_pos_tags

        self.embeddingph = embedding_placeholder
        self.embeddingph_size = self.embeddingph.get_shape()[0].value

        self.position_embedding_size = position_embedding_size
        self.pos_tag_embedding_size = pos_tag_embedding_size
        self.pos_tag_size = data_pos_tags.get_shape()[0].value

        self.filter_size = filter_size
        self.hidden_unit_size = hidden_unit_size
        self.learning_rate = learning_rate
        self.vocab_size = vocabulary_size
        self.embedding_size = embedding_size
        self.max_seq_len = self.data.get_shape()[1].value
        self.batch_size = self.data.get_shape()[0].value
        self.target_size = self.target.get_shape()[1].value

        fc_input_size = 3*self.filter_size
        print("KimCNN model filter out: {}".format(fc_input_size))

        # network variables (weights, biases, embeddings)
        self.weights = {
            # 1st convolution layer weights
            'fc1': tf.Variable(tf.random_normal([2, embedding_size+2*self.position_embedding_size+
                                                 self.pos_tag_embedding_size,
                                                 1, self.filter_size])),
            # 1st convolution layer weights
            'fc2': tf.Variable(tf.random_normal([3, embedding_size+2*self.position_embedding_size+
                                                 self.pos_tag_embedding_size,
                                                 1, self.filter_size])),
            # 1st convolution layer weights
            'fc3': tf.Variable(tf.random_normal([4, embedding_size+2*self.position_embedding_size+
                                                 self.pos_tag_embedding_size,
                                                 1, self.filter_size])),
            # perceptron layer weights
            'wd1': tf.Variable(tf.random_normal([fc_input_size, self.hidden_unit_size])),
            # output layer weights
            'out': tf.Variable(tf.random_normal([self.hidden_unit_size, self.target_size]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([self.filter_size])),
            'bc2': tf.Variable(tf.random_normal([self.filter_size])),
            'bc3': tf.Variable(tf.random_normal([self.filter_size])),
            'bd1': tf.Variable(tf.random_normal([hidden_unit_size])),
            'out': tf.Variable(tf.random_normal([self.target_size]))
        }

        # with tf.device('/cpu:0'):
        self.embedding_v_1 = tf.Variable(tf.zeros([self.embeddingph_size, self.embedding_size]),
                                         trainable=True,
                                         name="word_embedding_variable1",
                                         dtype=tf.float32)
        self.embedding_v_2 = tf.Variable(tf.zeros([self.embeddingph_size, self.embedding_size]),
                                         trainable=True,
                                         name="word_embedding_variable2",
                                         dtype=tf.float32)
        self.embedding_v_3 = tf.Variable(tf.zeros([self.embeddingph_size, self.embedding_size]),
                                         trainable=True,
                                         name="word_embedding_variable3",
                                         dtype=tf.float32)
        self.embedding_v_4 = tf.Variable(tf.zeros([self.embeddingph_size, self.embedding_size]),
                                         trainable=True,
                                         name="word_embedding_variable4",
                                         dtype=tf.float32)
        self.embedding_v_5 = tf.Variable(tf.zeros([self.embeddingph_size, self.embedding_size]),
                                         trainable=True,
                                         name="word_embedding_variable5",
                                         dtype=tf.float32)
        self.embedding_v_6 = tf.Variable(tf.zeros([self.embeddingph_size, self.embedding_size]),
                                         trainable=True,
                                         name="word_embedding_variable6",
                                         dtype=tf.float32)
        self.embedding_v_7 = tf.Variable(tf.zeros([self.embeddingph_size, self.embedding_size]),
                                         trainable=True,
                                         name="word_embedding_variable7",
                                         dtype=tf.float32)
        self.embedding_v_8 = tf.Variable(tf.zeros([self.embeddingph_size, self.embedding_size]),
                                         trainable=True,
                                         name="word_embedding_variable8",
                                         dtype=tf.float32)

        self.distance_chemical_embedding = tf.Variable(tf.random_normal([max_position_distance, self.position_embedding_size]),
                                                       trainable=True,
                                                       name="chemical_distance_variable",
                                                       dtype=tf.float32)
        self.distance_protein_embedding = tf.Variable(tf.random_normal([max_position_distance, self.position_embedding_size]),
                                                      trainable=True,
                                                      name="protein_distance_variable",
                                                      dtype=tf.float32)
        self.pos_tag_embedding = tf.Variable(tf.random_normal([self.pos_tag_size, self.pos_tag_embedding_size]),
                                             trainable=True,
                                             name="protein_distance_variable",
                                             dtype=tf.float32)

        self.assign1
        self.assign2
        self.assign3
        self.assign4
        self.assign5
        self.assign6
        self.assign7
        self.assign8

        self.prediction
        self.optimize
        self.error

    @staticmethod
    def conv2d(data, conv_filter_weights, b):
        data = tf.nn.conv2d(data, conv_filter_weights, strides=[1, 1, 1, 1],
                            padding='VALID')
        data = tf.nn.bias_add(data, b)
        return tf.nn.relu(data)

    @staticmethod
    def maxpool2d(data, k1, k2):
        return tf.nn.max_pool(data, ksize=[1, k1, k2, 1], strides=[1, k1, k2, 1], padding='VALID')

    @lazy_property
    def assign1(self):
        word_embedding_init1 = self.embedding_v_1.assign(self.embeddingph)
        return word_embedding_init1

    @lazy_property
    def assign2(self):
        word_embedding_init2 = self.embedding_v_2.assign(self.embeddingph)
        return word_embedding_init2

    @lazy_property
    def assign3(self):
        word_embedding_init3 = self.embedding_v_3.assign(self.embeddingph)
        return word_embedding_init3

    @lazy_property
    def assign4(self):
        word_embedding_init4 = self.embedding_v_4.assign(self.embeddingph)
        return word_embedding_init4

    @lazy_property
    def assign5(self):
        word_embedding_init5 = self.embedding_v_5.assign(self.embeddingph)
        return word_embedding_init5

    @lazy_property
    def assign6(self):
        word_embedding_init6 = self.embedding_v_6.assign(self.embeddingph)
        return word_embedding_init6

    @lazy_property
    def assign7(self):
        word_embedding_init7 = self.embedding_v_7.assign(self.embeddingph)
        return word_embedding_init7

    @lazy_property
    def assign8(self):
        word_embedding_init8 = self.embedding_v_8.assign(self.embeddingph)
        return word_embedding_init8

    @lazy_property
    def prediction(self):
        # get data and reshape it w.r.t network
        # embedding_data = tf.nn.embedding_lookup(self.embedding_v, self.data)
        embedding_data = tf.nn.embedding_lookup(params=[self.embedding_v_1, self.embedding_v_2, self.embedding_v_3,
                                                        self.embedding_v_4, self.embedding_v_5, self.embedding_v_6,
                                                        self.embedding_v_7, self.embedding_v_8],
                                                ids=self.data, partition_strategy='div')

        data = tf.reshape(embedding_data, [self.batch_size, self.max_seq_len, self.embedding_size, 1])

        protein_distance_embedding = tf.nn.embedding_lookup(params=self.distance_protein_embedding,
                                                            ids=self.distance_to_protein)

        protein_distance_data = tf.reshape(protein_distance_embedding, [self.batch_size, self.max_seq_len, self.position_embedding_size, 1])

        chemical_distance_embedding = tf.nn.embedding_lookup(params=self.distance_chemical_embedding,
                                                             ids=self.distance_to_chemical)

        chemical_distance_data = tf.reshape(chemical_distance_embedding,
                                            [self.batch_size, self.max_seq_len, self.position_embedding_size, 1])

        pos_tag_embedding = tf.nn.embedding_lookup(params=self.pos_tag_embedding,
                                                   ids=self.pos_tags)

        pos_tag_data = tf.reshape(pos_tag_embedding,
                                  [self.batch_size, self.max_seq_len, self.pos_tag_embedding_size, 1])

        data = tf.concat([data, protein_distance_data, chemical_distance_data, pos_tag_data], 2)

        # 1st convolution layer
        conv1 = self.conv2d(data, self.weights['fc1'], self.biases['bc1'])
        # 1st max pooling layer
        conv1 = self.maxpool2d(conv1, 143, 1)

        # 2st convolution layer
        conv2 = self.conv2d(data, self.weights['fc2'], self.biases['bc2'])
        # 2st max pooling layer
        conv2 = self.maxpool2d(conv2, 142, 1)

        # 3st convolution layer
        conv3 = self.conv2d(data, self.weights['fc3'], self.biases['bc3'])
        # 3st max pooling layer
        conv3 = self.maxpool2d(conv3, 141, 1)

        # fully connected layer
        # reshape conv2 data for fully connected layer
        fc1 = tf.concat([conv1, conv2, conv3], 3)
        fc1 = tf.reshape(fc1, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)

        # apply dropout
        fc1 = tf.nn.dropout(fc1, rate=0.5)

        # output, class predictions
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
        return out

    @lazy_property
    def optimize(self):
        unweighted_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction,
                                                                                       labels=self.target))
        loss = tf.reduce_mean(unweighted_loss_op)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(loss)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.cast(mistakes, tf.float32)
