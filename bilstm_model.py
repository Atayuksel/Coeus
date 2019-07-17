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
    def __init__(self, data, target, seq_lens, learning_rate, embedding_dimension, vocab_size, embedding_placeholder,
                 lstm_hidden_unit_size, distance_protein, distance_chemical, max_distance, position_embedding_size,
                 data_pos_tags, pos_tag_embedding_size, data_iob_tags, iob_tag_embedding_size,
                 input_fc_hidden_unit_size, embedding_chunk_size, input_representation):

        self.data = data
        self.target = target
        self.seq_lens = seq_lens
        self.distance_to_protein = distance_protein
        self.distance_to_chemical = distance_chemical
        self.input_representation = input_representation

        self.data_pos_tags = data_pos_tags
        self.pos_tag_embedding_size = pos_tag_embedding_size
        self.pos_tag_size = data_pos_tags.get_shape()[0].value

        self.data_iob_tags = data_iob_tags
        self.iob_tag_embedding_size = iob_tag_embedding_size
        self.iob_tag_size = data_iob_tags.get_shape()[0].value

        self.position_embedding_size = position_embedding_size
        self.learning_rate = learning_rate
        self.embedding_length = embedding_dimension
        self.vocabulary_size = vocab_size
        self.lstm_hidden_unit_size = lstm_hidden_unit_size

        self.embedding_placeholder = embedding_placeholder
        self.embedding_matrix_size = self.embedding_placeholder.get_shape()[0].value

        self.max_seq_len = self.data.get_shape()[1].value
        self.batch_size = self.data.get_shape()[0].value
        target_size = self.target.get_shape()[1].value

        self.weights = {
            # 'in': tf.Variable(tf.random_normal([ embedding_dimension + 2*position_embedding_size + pos_tag_embedding_size +  iob_tag_embedding_size, input_fc_hidden_unit_size])),
            'out': tf.Variable(tf.random_normal([2*self.lstm_hidden_unit_size, target_size]))}
        self.biases = {
            'out': tf.Variable(tf.random_normal([target_size]))
            # 'in': tf.Variable(tf.random_normal([input_fc_hidden_unit_size]))
            }

        # Embedding Variables
        self.embedding_chunk_list = []
        if embedding_chunk_size > 0:
            self.embedding_v_1 = tf.Variable(tf.zeros([self.embedding_matrix_size, self.embedding_length]),
                                             trainable=False,
                                             name="word_embedding_variable1",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_1)
        if embedding_chunk_size > 1:
            self.embedding_v_2 = tf.Variable(tf.zeros([self.embedding_matrix_size, self.embedding_length]),
                                             trainable=False,
                                             name="word_embedding_variable2",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_2)
        if embedding_chunk_size > 2:
            self.embedding_v_3 = tf.Variable(tf.zeros([self.embedding_matrix_size, self.embedding_length]),
                                             trainable=False,
                                             name="word_embedding_variable3",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_3)
        if embedding_chunk_size > 3:
            self.embedding_v_4 = tf.Variable(tf.zeros([self.embedding_matrix_size, self.embedding_length]),
                                             trainable=False,
                                             name="word_embedding_variable4",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_4)
        if embedding_chunk_size > 4:
            self.embedding_v_5 = tf.Variable(tf.zeros([self.embedding_matrix_size, self.embedding_length]),
                                             trainable=False,
                                             name="word_embedding_variable5",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_5)
        if embedding_chunk_size > 5:
            self.embedding_v_6 = tf.Variable(tf.zeros([self.embedding_matrix_size, self.embedding_length]),
                                             trainable=False,
                                             name="word_embedding_variable6",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_6)
        if embedding_chunk_size > 6:
            self.embedding_v_7 = tf.Variable(tf.zeros([self.embedding_matrix_size, self.embedding_length]),
                                             trainable=False,
                                             name="word_embedding_variable7",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_7)
        if embedding_chunk_size > 7:
            self.embedding_v_8 = tf.Variable(tf.zeros([self.embedding_matrix_size, self.embedding_length]),
                                             trainable=False,
                                             name="word_embedding_variable8",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_8)
        if embedding_chunk_size > 8:
            self.embedding_v_9 = tf.Variable(tf.zeros([self.embedding_matrix_size, self.embedding_length]),
                                             trainable=False,
                                             name="word_embedding_variable9",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_9)

        self.distance_chemical_embedding = tf.Variable(tf.random_normal([max_distance, self.position_embedding_size]),
                                                       trainable=True,
                                                       name="chemical_distance_variable",
                                                       dtype=tf.float32)
        self.distance_protein_embedding = tf.Variable(tf.random_normal([max_distance, self.position_embedding_size]),
                                                      trainable=True,
                                                      name="protein_distance_variable",
                                                      dtype=tf.float32)
        self.pos_tag_embedding = tf.Variable(tf.random_normal([self.pos_tag_size, self.pos_tag_embedding_size]),
                                             trainable=True,
                                             name="pos_tag_embedding_variable",
                                             dtype=tf.float32)
        self.iob_tag_embedding = tf.Variable(tf.random_normal([self.iob_tag_size, self.iob_tag_embedding_size]),
                                             trainable=True,
                                             name="iob_tag_embedding_variable",
                                             dtype=tf.float32)

        if embedding_chunk_size > 0:
            self.assign1
        if embedding_chunk_size > 1:
            self.assign2
        if embedding_chunk_size > 2:
            self.assign3
        if embedding_chunk_size > 3:
            self.assign4
        if embedding_chunk_size > 4:
            self.assign5
        if embedding_chunk_size > 5:
            self.assign6
        if embedding_chunk_size > 6:
            self.assign7
        if embedding_chunk_size > 7:
            self.assign8
        if embedding_chunk_size > 8:
            self.assign9

        self.prediction
        self.optimize

    @lazy_property
    def assign1(self):
        word_embedding_init1 = self.embedding_v_1.assign(self.embedding_placeholder)
        return word_embedding_init1

    @lazy_property
    def assign2(self):
        word_embedding_init2 = self.embedding_v_2.assign(self.embedding_placeholder)
        return word_embedding_init2

    @lazy_property
    def assign3(self):
        word_embedding_init3 = self.embedding_v_3.assign(self.embedding_placeholder)
        return word_embedding_init3

    @lazy_property
    def assign4(self):
        word_embedding_init4 = self.embedding_v_4.assign(self.embedding_placeholder)
        return word_embedding_init4

    @lazy_property
    def assign5(self):
        word_embedding_init5 = self.embedding_v_5.assign(self.embedding_placeholder)
        return word_embedding_init5

    @lazy_property
    def assign6(self):
        word_embedding_init6 = self.embedding_v_6.assign(self.embedding_placeholder)
        return word_embedding_init6

    @lazy_property
    def assign7(self):
        word_embedding_init7 = self.embedding_v_7.assign(self.embedding_placeholder)
        return word_embedding_init7

    @lazy_property
    def assign8(self):
        word_embedding_init8 = self.embedding_v_8.assign(self.embedding_placeholder)
        return word_embedding_init8

    @lazy_property
    def assign9(self):
        word_embedding_init9 = self.embedding_v_9.assign(self.embedding_placeholder)
        return word_embedding_init9

    def bilstm_layer(self, data, keep_prob):
        max_seq_len = data.get_shape()[1]
        x = tf.unstack(data, max_seq_len, 1)
        lstm_fw_cell = rnn.LSTMBlockCell(num_units=self.lstm_hidden_unit_size)
        lstm_fw_cell_dropout = rnn.DropoutWrapper(cell=lstm_fw_cell,
                                                  input_keep_prob=keep_prob,
                                                  output_keep_prob=keep_prob,
                                                  state_keep_prob=keep_prob)
        lstm_bw_cell = rnn.LSTMBlockCell(num_units=self.lstm_hidden_unit_size)
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
        embedding_data = tf.nn.embedding_lookup(params=self.embedding_chunk_list,
                                                ids=self.data, partition_strategy='div')

        protein_distance_embedding = tf.nn.embedding_lookup(params=self.distance_protein_embedding,
                                                            ids=self.distance_to_protein)

        chemical_distance_embedding = tf.nn.embedding_lookup(params=self.distance_chemical_embedding,
                                                             ids=self.distance_to_chemical)

        batch_pos_tag_embedding = tf.nn.embedding_lookup(params=self.pos_tag_embedding,
                                                         ids=self.data_pos_tags)

        batch_iob_tag_embedding = tf.nn.embedding_lookup(params=self.iob_tag_embedding,
                                                         ids=self.data_iob_tags)

        if self.input_representation == 0:
            data = embedding_data
        elif self.input_representation == 1:
            data = tf.concat([embedding_data, protein_distance_embedding, chemical_distance_embedding], 2)
        elif self.input_representation == 2:
            data = tf.concat([embedding_data, protein_distance_embedding, chemical_distance_embedding,
                              batch_pos_tag_embedding], 2)
        elif self.input_representation == 3:
            data = tf.concat([embedding_data, protein_distance_embedding, chemical_distance_embedding,
                              batch_pos_tag_embedding, batch_iob_tag_embedding], 2)

        # data = tf.reshape(data, [-1, self.weights['in'].get_shape().as_list()[0]])
        # data = tf.add(tf.matmul(data, self.weights['in']), self.biases['in'])
        # data = tf.nn.relu(data)
        # data = tf.reshape(data, [self.batch_size, self.max_seq_len, -1])

        rnn_output = self.bilstm_layer(data, 0.5)
        max_pooling_output = self.lstm_max_pooling(rnn_output)
        logits = tf.add(tf.matmul(max_pooling_output, self.weights['out']), self.biases['out'])
        return logits

    @lazy_property
    def optimize(self):
        unweighted_loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction, labels=self.target)
        loss = tf.reduce_mean(unweighted_loss_op)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(loss)
