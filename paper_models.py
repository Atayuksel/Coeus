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
    def __init__(self, word_ids_placeholder, distance_protein_placeholder, distance_chemical_placeholder,
                 pos_tag_placeholder, iob_tag_placeholder, label_placeholder, word_embedding_placeholder,
                 position_embedding_placeholder, pos_tag_embedding_placeholder, iob_tag_embedding_placeholder,
                 position_embedding_flag, pos_tag_embedding_flag, iob_tag_embedding_flag, word_embedding_chunk_number,
                 learning_rate, hidden_unit_size, filter_size):

        # data placeholder
        self.word_ids_placeholder = word_ids_placeholder
        self.distance_to_protein_placeholder = distance_protein_placeholder
        self.distance_to_chemical_placeholder = distance_chemical_placeholder
        self.pos_tag_placeholder = pos_tag_placeholder
        self.iob_tag_placeholder = iob_tag_placeholder

        # label placeholder
        self.label_placeholder = label_placeholder

        # embedding placeholder
        self.word_embedding_placeholder = word_embedding_placeholder
        self.position_embedding_placeholder = position_embedding_placeholder
        self.pos_tag_embedding_placeholder = pos_tag_embedding_placeholder
        self.iob_tag_embedding_placeholder = iob_tag_embedding_placeholder

        # embedding flag
        self.position_embedding_flag = position_embedding_flag
        self.pos_tag_embedding_flag = pos_tag_embedding_flag
        self.iob_tag_embedding_flag = iob_tag_embedding_flag

        # embedding sizes
        self.position_tag_size = position_embedding_placeholder.get_shape()[0].value
        self.pos_tag_size = pos_tag_embedding_placeholder.get_shape()[0].value
        self.iob_tag_size = iob_tag_embedding_placeholder.get_shape()[0].value

        self.position_embedding_size = position_embedding_placeholder.get_shape()[1].value
        self.pos_tag_embedding_size = pos_tag_embedding_placeholder.get_shape()[1].value
        self.iob_tag_embedding_size = iob_tag_embedding_placeholder.get_shape()[1].value

        # word embedding
        self.word_embedding_chunk_number = word_embedding_chunk_number
        self.vocabulary_size = word_embedding_placeholder.get_shape()[0].value
        self.word_embedding_size = word_embedding_placeholder.get_shape()[1].value

        # model hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = word_ids_placeholder.get_shape()[0].value
        self.target_size = self.label_placeholder.get_shape()[1].value
        self.hidden_unit_size = hidden_unit_size
        self.filter_size = filter_size
        self.fcl_input_size = 3*self.filter_size
        self.max_seq_len = word_ids_placeholder.get_shape()[1].value

        self.total_embedding_size = self.word_embedding_size
        if position_embedding_flag:
            self.total_embedding_size = self.total_embedding_size + 2*self.position_embedding_size
        if pos_tag_embedding_flag:
            self.total_embedding_size = self.total_embedding_size + self.pos_tag_embedding_size
        if iob_tag_embedding_flag:
            self.total_embedding_size = self.total_embedding_size + self.iob_tag_embedding_size

        # network variables (weights, biases, embeddings)
        self.weights = {
            # 1st convolution layer weights
            'fc1': tf.Variable(tf.random_normal([2, self.total_embedding_size,
                                                 1, self.filter_size])),
            # 1st convolution layer weights
            'fc2': tf.Variable(tf.random_normal([3, self.total_embedding_size,
                                                 1, self.filter_size])),
            # 1st convolution layer weights
            'fc3': tf.Variable(tf.random_normal([4, self.total_embedding_size,
                                                 1, self.filter_size])),
            # perceptron layer weights
            'wd1': tf.Variable(tf.random_normal([self.fcl_input_size, self.hidden_unit_size])),
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

        # Embedding Variables
        self.embedding_chunk_list = []
        if self.word_embedding_chunk_number > 0:
            self.embedding_v_1 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=False,
                                             name="word_embedding_variable1",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_1)

        if self.word_embedding_chunk_number > 1:
            self.embedding_v_2 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=False,
                                             name="word_embedding_variable2",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_2)

        if self.word_embedding_chunk_number > 2:
            self.embedding_v_3 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=False,
                                             name="word_embedding_variable3",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_3)

        if self.word_embedding_chunk_number > 3:
            self.embedding_v_4 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=False,
                                             name="word_embedding_variable4",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_4)

        if self.word_embedding_chunk_number > 4:
            self.embedding_v_5 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=False,
                                             name="word_embedding_variable5",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_5)

        if self.word_embedding_chunk_number > 5:
            self.embedding_v_6 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=False,
                                             name="word_embedding_variable6",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_6)

        if self.word_embedding_chunk_number > 6:
            self.embedding_v_7 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=False,
                                             name="word_embedding_variable7",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_7)

        if self.word_embedding_chunk_number > 7:
            self.embedding_v_8 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=False,
                                             name="word_embedding_variable8",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_8)

        if self.word_embedding_chunk_number > 8:
            self.embedding_v_9 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=False,
                                             name="word_embedding_variable9",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_9)

        if self.position_embedding_flag:
            self.distance_chemical_embedding = tf.Variable(tf.random_normal([self.position_tag_size,
                                                                             self.position_embedding_size]),
                                                           trainable=True,
                                                           name="chemical_distance_variable",
                                                           dtype=tf.float32)

        if self.position_embedding_flag:
            self.distance_protein_embedding = tf.Variable(tf.random_normal([self.position_tag_size,
                                                                            self.position_embedding_size]),
                                                          trainable=True,
                                                          name="protein_distance_variable",
                                                          dtype=tf.float32)

        if self.pos_tag_embedding_flag:
            self.pos_tag_embedding = tf.Variable(tf.random_normal([self.pos_tag_size,
                                                                   self.pos_tag_embedding_size]),
                                                 trainable=True,
                                                 name="pos_tag_embedding_variable",
                                                 dtype=tf.float32)

        if self.iob_tag_embedding_flag:
            self.iob_tag_embedding = tf.Variable(tf.random_normal([self.iob_tag_size,
                                                                   self.iob_tag_embedding_size]),
                                                 trainable=True,
                                                 name="iob_tag_embedding_variable",
                                                 dtype=tf.float32)

        if self.word_embedding_chunk_number > 0:
            self.assign1
        if self.word_embedding_chunk_number > 1:
            self.assign2
        if self.word_embedding_chunk_number > 2:
            self.assign3
        if self.word_embedding_chunk_number > 3:
            self.assign4
        if self.word_embedding_chunk_number > 4:
            self.assign5
        if self.word_embedding_chunk_number > 5:
            self.assign6
        if self.word_embedding_chunk_number > 6:
            self.assign7
        if self.word_embedding_chunk_number > 7:
            self.assign8
        if self.word_embedding_chunk_number > 8:
            self.assign9

        self.assign_chemical_position_embeddings
        self.assign_protein_position_embeddings
        self.assign_pos_tag_embeddings
        self.assign_iob_tag_embeddings
        self.prediction
        self.optimize

    @lazy_property
    def assign_chemical_position_embeddings(self):
        chemical_position_embedding_assignment = self.distance_chemical_embedding.assign(self.position_embedding_placeholder)
        return chemical_position_embedding_assignment

    @lazy_property
    def assign_protein_position_embeddings(self):
        protein_position_embedding_assignment = self.distance_protein_embedding.assign(self.position_embedding_placeholder)
        return protein_position_embedding_assignment

    @lazy_property
    def assign_pos_tag_embeddings(self):
        pos_tag_assignment = self.pos_tag_embedding.assign(self.pos_tag_embedding_placeholder)
        return pos_tag_assignment

    @lazy_property
    def assign_iob_tag_embeddings(self):
        iob_tag_assignment = self.iob_tag_embedding.assign(self.iob_tag_embedding_placeholder)
        return iob_tag_assignment

    @lazy_property
    def assign1(self):
        word_embedding_init1 = self.embedding_v_1.assign(self.word_embedding_placeholder)
        return word_embedding_init1

    @lazy_property
    def assign2(self):
        word_embedding_init2 = self.embedding_v_2.assign(self.word_embedding_placeholder)
        return word_embedding_init2

    @lazy_property
    def assign3(self):
        word_embedding_init3 = self.embedding_v_3.assign(self.word_embedding_placeholder)
        return word_embedding_init3

    @lazy_property
    def assign4(self):
        word_embedding_init4 = self.embedding_v_4.assign(self.word_embedding_placeholder)
        return word_embedding_init4

    @lazy_property
    def assign5(self):
        word_embedding_init5 = self.embedding_v_5.assign(self.word_embedding_placeholder)
        return word_embedding_init5

    @lazy_property
    def assign6(self):
        word_embedding_init6 = self.embedding_v_6.assign(self.word_embedding_placeholder)
        return word_embedding_init6

    @lazy_property
    def assign7(self):
        word_embedding_init7 = self.embedding_v_7.assign(self.word_embedding_placeholder)
        return word_embedding_init7

    @lazy_property
    def assign8(self):
        word_embedding_init8 = self.embedding_v_8.assign(self.word_embedding_placeholder)
        return word_embedding_init8

    @lazy_property
    def assign9(self):
        word_embedding_init9 = self.embedding_v_9.assign(self.word_embedding_placeholder)
        return word_embedding_init9

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
    def prediction(self):
        # get data and reshape it w.r.t network
        # embedding_data = tf.nn.embedding_lookup(self.embedding_v, self.data)
        word_embedding_data = tf.nn.embedding_lookup(params=self.embedding_chunk_list,
                                                     ids=self.data_placeholder, partition_strategy='div')
        word_embedding_data = tf.reshape(word_embedding_data, [self.batch_size,
                                                               self.max_seq_length,
                                                               self.word_embedding_size,
                                                               1])
        data = word_embedding_data
        if self.position_embedding_flag:
            protein_distance_embedding = tf.nn.embedding_lookup(params=self.distance_protein_embedding,
                                                                ids=self.distance_protein_placeholder)
            protein_distance_data = tf.reshape(protein_distance_embedding, [self.batch_size,
                                                                            self.max_seq_length,
                                                                            self.position_embedding_size,
                                                                            1])

            chemical_distance_embedding = tf.nn.embedding_lookup(params=self.distance_chemical_embedding,
                                                                 ids=self.distance_chemical_placeholder)
            chemical_distance_data = tf.reshape(chemical_distance_embedding, [self.batch_size,
                                                                              self.max_seq_length,
                                                                              self.position_embedding_size,
                                                                              1])
            data = tf.concat([data, protein_distance_data, chemical_distance_data], 2)

        if self.pos_tag_embedding_flag:
            pos_tag_embedding = tf.nn.embedding_lookup(params=self.pos_tag_embedding,
                                                       ids=self.pos_tag_placeholder)
            pos_tag_data = tf.reshape(pos_tag_embedding, [self.batch_size,
                                                          self.max_seq_length,
                                                          self.pos_tag_embedding_size,
                                                          1])
            data = tf.concat([data, pos_tag_data], 2)

        if self.iob_tag_embedding_flag:
            iob_tag_embedding = tf.nn.embedding_lookup(params=self.iob_tag_embedding,
                                                       ids=self.iob_tag_placeholder)
            iob_tag_data = tf.reshape(iob_tag_embedding, [self.batch_size,
                                                          self.max_seq_length,
                                                          self.iob_tag_embedding_size,
                                                          1])
            data = tf.concat([data, iob_tag_data], 2)

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
                                                                                       labels=self.label_placeholder))
        loss = tf.reduce_mean(unweighted_loss_op)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(loss)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.label_placeholder, 1), tf.argmax(self.prediction, 1))
        return tf.cast(mistakes, tf.float32)
