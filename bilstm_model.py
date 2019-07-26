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
    def __init__(self, word_ids_placeholder, seq_length_placeholder, distance_protein_placeholder,
                 distance_chemical_placeholder, pos_tag_placeholder, iob_tag_placeholder, label_placeholder,
                 word_embedding_placeholder, position_embedding_placeholder, pos_tag_embedding_placeholder,
                 iob_tag_embedding_placeholder, position_embedding_flag, pos_tag_embedding_flag, iob_tag_embedding_flag,
                 word_embedding_chunk_number, learning_rate, lstm_hidden_unit_size, fcl_hidden_unit_size,
                 train_word_embeddings):

        # data placeholder
        self.word_ids_placeholder = word_ids_placeholder
        self.seq_length_placeholder = seq_length_placeholder
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
        if self.position_embedding_flag:
            self.position_tag_size = position_embedding_placeholder.get_shape()[0].value
            self.position_embedding_size = position_embedding_placeholder.get_shape()[1].value

        if self.pos_tag_embedding_flag:
            self.pos_tag_size = pos_tag_embedding_placeholder.get_shape()[0].value
            self.pos_tag_embedding_size = pos_tag_embedding_placeholder.get_shape()[1].value

        if self.iob_tag_embedding_flag:
            self.iob_tag_size = iob_tag_embedding_placeholder.get_shape()[0].value
            self.iob_tag_embedding_size = iob_tag_embedding_placeholder.get_shape()[1].value

        # word embedding
        self.word_embedding_chunk_number = word_embedding_chunk_number
        self.vocabulary_size = word_embedding_placeholder.get_shape()[0].value
        self.word_embedding_size = word_embedding_placeholder.get_shape()[1].value
        self.train_word_embeddings = train_word_embeddings

        # model hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = word_ids_placeholder.get_shape()[0].value
        self.target_size = self.label_placeholder.get_shape()[1].value
        self.lstm_hidden_unit_size = lstm_hidden_unit_size
        self.fcl_hidden_unit_size = fcl_hidden_unit_size
        self.max_seq_len = word_ids_placeholder.get_shape()[1].value

        if self.fcl_hidden_unit_size != 0:
            fcl_input_size = self.word_embedding_size
            if self.position_embedding_flag:
                fcl_input_size = fcl_input_size + 2*self.position_embedding_size
            if self.pos_tag_embedding_flag:
                fcl_input_size = fcl_input_size + self.pos_tag_embedding_size
            if self.iob_tag_embedding_flag:
                fcl_input_size = fcl_input_size + self.iob_tag_embedding_size
        else:
            fcl_input_size = 0

        self.weights = {
            'in': tf.Variable(tf.random_normal([fcl_input_size, self.fcl_hidden_unit_size])),
            'out': tf.Variable(tf.random_normal([2*self.lstm_hidden_unit_size, self.target_size]))
        }

        self.biases = {
            'in': tf.Variable(tf.random_normal([self.fcl_hidden_unit_size])),
            'out': tf.Variable(tf.random_normal([self.target_size]))
        }

        # Embedding Variables
        self.embedding_chunk_list = []
        if self.word_embedding_chunk_number > 0:
            self.embedding_v_1 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=self.train_word_embeddings,
                                             name="word_embedding_variable1",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_1)

        if self.word_embedding_chunk_number > 1:
            self.embedding_v_2 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=self.train_word_embeddings,
                                             name="word_embedding_variable2",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_2)

        if self.word_embedding_chunk_number > 2:
            self.embedding_v_3 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=self.train_word_embeddings,
                                             name="word_embedding_variable3",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_3)

        if self.word_embedding_chunk_number > 3:
            self.embedding_v_4 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=self.train_word_embeddings,
                                             name="word_embedding_variable4",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_4)

        if self.word_embedding_chunk_number > 4:
            self.embedding_v_5 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=self.train_word_embeddings,
                                             name="word_embedding_variable5",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_5)

        if self.word_embedding_chunk_number > 5:
            self.embedding_v_6 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=self.train_word_embeddings,
                                             name="word_embedding_variable6",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_6)

        if self.word_embedding_chunk_number > 6:
            self.embedding_v_7 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=self.train_word_embeddings,
                                             name="word_embedding_variable7",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_7)

        if self.word_embedding_chunk_number > 7:
            self.embedding_v_8 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=self.train_word_embeddings,
                                             name="word_embedding_variable8",
                                             dtype=tf.float32)
            self.embedding_chunk_list.append(self.embedding_v_8)

        if self.word_embedding_chunk_number > 8:
            self.embedding_v_9 = tf.Variable(tf.zeros([self.vocabulary_size, self.word_embedding_size]),
                                             trainable=self.train_word_embeddings,
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

        if self.position_embedding_flag:
            self.assign_chemical_position_embeddings
            self.assign_protein_position_embeddings

        if self.pos_tag_embedding_flag:
            self.assign_pos_tag_embeddings

        if self.iob_tag_embedding_flag:
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

    def bilstm_layer(self, data, keep_prob):
        x = tf.unstack(data, self.max_seq_len, 1)
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
                                                        sequence_length=self.seq_length_placeholder,
                                                        dtype=tf.float32)
        return rnn_output

    def lstm_max_pooling(self, rnn_output):
        lstm_output = tf.stack(values=rnn_output, axis=1)
        result = []
        for i in range(lstm_output.shape[0].value):
            sentence_length = self.seq_length_placeholder[i]
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
        word_embedding_data = tf.nn.embedding_lookup(params=self.embedding_chunk_list,
                                                     ids=self.word_ids_placeholder, partition_strategy='div')
        data = word_embedding_data

        if self.position_embedding_flag:
            protein_distance_embedding = tf.nn.embedding_lookup(params=self.distance_protein_embedding,
                                                                ids=self.distance_to_protein_placeholder)
            chemical_distance_embedding = tf.nn.embedding_lookup(params=self.distance_chemical_embedding,
                                                                 ids=self.distance_to_chemical_placeholder)
            data = tf.concat([data, protein_distance_embedding, chemical_distance_embedding], 2)

        if self.pos_tag_embedding_flag:
            pos_tag_embedding = tf.nn.embedding_lookup(params=self.pos_tag_embedding,
                                                       ids=self.pos_tag_placeholder)
            data = tf.concat([data, pos_tag_embedding], 2)

        if self.iob_tag_embedding_flag:
            iob_tag_embedding = tf.nn.embedding_lookup(params=self.iob_tag_embedding,
                                                       ids=self.iob_tag_placeholder)
            data = tf.concat([data, iob_tag_embedding], 2)

        if self.fcl_hidden_unit_size != 0:
            data = tf.reshape(data, [-1, self.weights['in'].get_shape().as_list()[0]])
            data = tf.add(tf.matmul(data, self.weights['in']), self.biases['in'])
            data = tf.nn.relu(data)
            data = tf.reshape(data, [self.batch_size, self.max_seq_len, -1])

        rnn_output = self.bilstm_layer(data, 0.5)
        max_pooling_output = self.lstm_max_pooling(rnn_output)
        logits = tf.add(tf.matmul(max_pooling_output, self.weights['out']), self.biases['out'])
        return logits

    @lazy_property
    def optimize(self):
        unweighted_loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.prediction,
                                                                        labels=self.label_placeholder)
        loss = tf.reduce_mean(unweighted_loss_op)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(loss)
