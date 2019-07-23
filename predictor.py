import datetime
import configparser
import numpy as np
import math
import tensorflow as tf
import bilstm_model as bilstm
import paper_models as papers
import progressbar
import pickle


class Predictor(object):
    def __init__(self, predictor_id, data_interface, report_directory,
                 development_set_flag, test_set_flag, early_stopping_set):

        # reset previously created tensorflow models.
        tf.reset_default_graph()

        # get class parameters
        self.predictor_id = predictor_id
        self.data_interface = data_interface
        self.report_directory = report_directory

        # boolean flags if prediction is applied in the dataset
        self.development_set_flag = development_set_flag
        self.test_set_flag = test_set_flag

        # which dataset to use for early stopping
        self.early_stopping_set = early_stopping_set

        # open configuration file
        config = configparser.ConfigParser()
        config.read('config.ini')

        # read model parameters
        section_name = 'MODEL'
        self.model_type = config[section_name]['model_type']
        self.num_epoch = int(config[section_name]['num_epoch'])
        self.learning_rate = float(config[section_name]['learning_rate'])
        self.error_function = config[section_name]['error_function']
        self.output_directory = config[section_name]['output_directory']

        # create report file
        self.report_name = str(self.predictor_id) + '_' + self.model_type + '_report.txt'
        self.report_name = self.report_directory + '/' + self.report_name
        self.report_file = open(self.report_name, "w+")

        # write basic information to the report file
        # time information
        line = "Time: {}".format(str(datetime.datetime.now()))
        print(line)
        self.report_file.write(line+'\n')
        # report file name
        line = "Report File Name: {}".format(str(self.report_name))
        print(line)
        self.report_file.write(line+'\n')
        # model type
        line = "Model: {}".format(self.model_type)
        print(line)
        self.report_file.write(line + '\n')
        # number of epoch
        line = "Number of Epoch: {}".format(str(self.num_epoch))
        print(line)
        self.report_file.write(line + "\n")
        # learning rate
        line = "Learning Rate: {}".format(str(self.learning_rate))
        print(line)
        self.report_file.write(line+'\n')
        # error function
        line = "Error Function: {}".format(str(self.error_function))
        print(line)
        self.report_file.write(line+'\n')
        # output directory
        line = "Error Function: {}".format(str(self.output_directory))
        print(line)
        self.report_file.write(line+'\n')

        # best metrics, early stopping
        self.best_epoch = 0

        if self.development_set_flag:
            self.best_development_metrics = [0, 0, 0]  # precision, recall, f1-measure
            self.best_development_predictions = []
            self.best_development_logits = []
            self.best_development_truth_values = []

        if self.test_set_flag:
            self.best_test_metrics = [0, 0, 0]  # precision, recall, f1-measure
            self.best_test_predictions = []
            self.best_test_logits = []
            self.best_test_truth_values = []

        # print data interface information
        self.data_interface.print_information()

        # write data interface information to report file
        self.data_interface.write_information(file=self.report_file)

        # get word embeddings
        self.word_embedding_matrix = self.data_interface.word_embedding_matrix
        self.vocabulary_size = self.word_embedding_matrix.shape[0]
        self.word_embedding_dimension = self.word_embedding_matrix.shape[1]

        # split word embedding matrix
        self.embedding_chunk_number = 0
        for i in range(10, 0, -1):
            if self.vocabulary_size % i == 0:
                self.embedding_chunk_number = i
                break
        self.word_embedding_matrices = np.split(self.word_embedding_matrix, self.embedding_chunk_number)

        # calculate number of steps in batch for each dataset
        self.training_num_batch = math.ceil(len(self.data_interface.dataset['training']['data'])
                                            / self.data_interface.batch_size)
        if self.development_set_flag:
            self.development_num_batch = math.ceil(len(self.data_interface.dataset['development']['data'])
                                                   / self.data_interface.batch_size)
        if self.test_set_flag:
            self.test_num_batch = math.ceil(len(self.data_interface.dataset['test']['data'])
                                            / self.data_interface.batch_size)

        # fetch embeddings information
        embeddings_info = self.data_interface.get_embedding_information()
        self.position_embedding_flag = embeddings_info['position_embedding_flag']
        if self.position_embedding_flag:
            self.position_embedding_size = embeddings_info['position_embedding_size']
            self.position_ids_size = embeddings_info['position_ids_size']
        self.pos_tag_embedding_flag = embeddings_info['pos_tag_embedding_flag']
        if self.pos_tag_embedding_flag:
            self.pos_tag_embedding_size = embeddings_info['pos_tag_embedding_size']
            self.pos_tag_ids_size = embeddings_info['pos_tag_ids_size']
        self.iob_tag_embedding_flag = embeddings_info['iob_tag_embedding_flag']
        if self.iob_tag_embedding_flag:
            self.iob_tag_embedding_size = embeddings_info['iob_tag_embedding_size']
            self.iob_tag_ids_size = embeddings_info['iob_tag_ids_size']

        # get embeddings
        if self.position_embedding_flag:
            self.position_embedding_matrix = self.data_interface.position_embedding_matrix
        if self.pos_tag_embedding_flag:
            self.pos_tag_embedding_matrix = self.data_interface.pos_tag_embedding_matrix
        if self.iob_tag_embedding_flag:
            self.iob_tag_embedding_matrix = self.data_interface.iob_tag_embedding_matrix

        # TensorFlow placeholders
        self.word_ids_tf_placeholder = tf.placeholder(tf.int64, [self.data_interface.batch_size,
                                                                 self.data_interface.dataset['training']['max_seq_len']],
                                                      name='data_placeholder')

        self.seq_len_tf_placeholder = tf.placeholder(tf.int64, [self.data_interface.batch_size, ],
                                                     name='seq_len_placeholder')

        self.distance_protein_tf_placeholder = tf.placeholder(tf.int64, [self.data_interface.batch_size,
                                                                         self.data_interface.dataset['training'][
                                                                             'max_seq_len']],
                                                              name='distance_protein_placeholder')

        self.distance_chemical_tf_placeholder = tf.placeholder(tf.int64, [self.data_interface.batch_size,
                                                                          self.data_interface.dataset['training'][
                                                                              'max_seq_len']],
                                                               name='distance_chemical_placeholder')

        self.pos_tag_tf_placeholder = tf.placeholder(tf.int64, [self.data_interface.batch_size,
                                                                self.data_interface.dataset['training']['max_seq_len']],
                                                     name='pos_tag_placeholder')

        self.iob_tag_tf_placeholder = tf.placeholder(tf.int64, [self.data_interface.batch_size,
                                                                self.data_interface.dataset['training']['max_seq_len']],
                                                     name='iob_tag_placeholder')

        self.label_tf_placeholder = tf.placeholder(tf.float32, [self.data_interface.batch_size, 2],
                                                   name='label_placeholder')

        self.word_embedding_tf_placeholder = tf.placeholder(tf.float32, [self.word_embedding_matrices[0].shape[0],
                                                                         self.word_embedding_dimension],
                                                            name='word_embedding_placeholder')

        if self.position_embedding_flag:
            self.position_embedding_tf_placeholder = tf.placeholder(tf.float32, [self.position_ids_size,
                                                                                 self.position_embedding_size],
                                                                    name='position_embedding_placeholder')
        else:
            self.position_embedding_tf_placeholder = None

        if self.pos_tag_embedding_flag:
            self.pos_tag_embedding_tf_placeholder = tf.placeholder(tf.float32, [self.pos_tag_ids_size,
                                                                                self.pos_tag_embedding_size],
                                                                    name='pos_tag_embedding_placeholder')
        else:
            self.pos_tag_embedding_tf_placeholder = None

        if self.iob_tag_embedding_flag:
            self.iob_tag_embedding_tf_placeholder = tf.placeholder(tf.float32, [self.iob_tag_ids_size,
                                                                                self.iob_tag_embedding_size],
                                                                   name='iob_tag_embedding_placeholder')
        else:
            self.iob_tag_embedding_tf_placeholder = None

        # TensorFlow models
        if self.model_type == 'bilstm':
            print('Model HyperParameters:')
            self.report_file.write('\n' + 'Model HyperParameters:' + '\n')

            # obtain hyper parameters specific to the model
            section_name = 'BILSTM'

            lstm_hidden_unit_size = int(config[section_name]['lstm_hidden_unit'])
            print('LSTM Hidden Unit Size: {}'.format(lstm_hidden_unit_size))
            self.report_file.write('LSTM Hidden Unit Size: {}'.format(lstm_hidden_unit_size)+'\n')

            fcl_hidden_unit_size = int(config[section_name]['lstm_in_hidden_unit'])
            print('BiLSTM FCL Hidden Unit Size: {}'.format(fcl_hidden_unit_size))
            self.report_file.write('BiLSTM FCL Hidden Unit Size: {}'.format(fcl_hidden_unit_size)+'\n')

            # create the model
            self.model = bilstm.BiLSTMModel(word_ids_placeholder=self.word_ids_tf_placeholder,
                                            seq_length_placeholder=self.seq_len_tf_placeholder,
                                            distance_protein_placeholder=self.distance_protein_tf_placeholder,
                                            distance_chemical_placeholder=self.distance_chemical_tf_placeholder,
                                            pos_tag_placeholder=self.pos_tag_tf_placeholder,
                                            iob_tag_placeholder=self.iob_tag_tf_placeholder,
                                            label_placeholder=self.label_tf_placeholder,
                                            word_embedding_placeholder=self.word_embedding_tf_placeholder,
                                            position_embedding_placeholder=self.position_embedding_tf_placeholder,
                                            pos_tag_embedding_placeholder=self.pos_tag_embedding_tf_placeholder,
                                            iob_tag_embedding_placeholder=self.iob_tag_embedding_tf_placeholder,
                                            position_embedding_flag=self.position_embedding_flag,
                                            pos_tag_embedding_flag=self.pos_tag_embedding_flag,
                                            iob_tag_embedding_flag=self.iob_tag_embedding_flag,
                                            word_embedding_chunk_number=self.embedding_chunk_number,
                                            learning_rate=self.learning_rate,
                                            lstm_hidden_unit_size=lstm_hidden_unit_size,
                                            fcl_hidden_unit_size=fcl_hidden_unit_size)

        elif self.model_type == 'cnn':
            print('Model HyperParameters:')
            self.report_file.write('\n' + 'Model HyperParameters:' + '\n')

            # obtain hyper parameters specific to the model
            section_name = 'CNN'

            cnn_filter_out = int(config[section_name]['cnn_filter_out'])
            print('CNN Filter Out: {}'.format(str(cnn_filter_out)))
            self.report_file.write('CNN Filter Out: {}'.format(str(cnn_filter_out))+'\n')

            fcl_hidden_unit_size = int(config[section_name]['cnn_hidden_unit'])
            print('CNN FCL Hidden Unit Size: {}'.format(str(fcl_hidden_unit_size)))
            self.report_file.write('CNN FCL Hidden Unit Size: {}'.format(str(fcl_hidden_unit_size))+'\n')

            self.model = papers.KimCNN(word_ids_placeholder=self.word_ids_tf_placeholder,
                                       distance_protein_placeholder=self.distance_protein_tf_placeholder,
                                       distance_chemical_placeholder=self.distance_chemical_tf_placeholder,
                                       pos_tag_placeholder=self.pos_tag_tf_placeholder,
                                       iob_tag_placeholder=self.iob_tag_tf_placeholder,
                                       label_placeholder=self.label_tf_placeholder,
                                       word_embedding_placeholder=self.word_embedding_tf_placeholder,
                                       position_embedding_placeholder=self.position_embedding_tf_placeholder,
                                       pos_tag_embedding_placeholder=self.pos_tag_embedding_tf_placeholder,
                                       iob_tag_embedding_placeholder=self.iob_tag_embedding_tf_placeholder,
                                       position_embedding_flag=self.position_embedding_flag,
                                       pos_tag_embedding_flag=self.pos_tag_embedding_flag,
                                       iob_tag_embedding_flag=self.iob_tag_embedding_flag,
                                       word_embedding_chunk_number=self.embedding_chunk_number,
                                       learning_rate=self.learning_rate,
                                       hidden_unit_size=fcl_hidden_unit_size,
                                       filter_size=cnn_filter_out)

    def train(self, min_epoch_number):
        with tf.Session() as sess:
            print("TF session is created successfully\n")

            # initialize variables
            sess.run(tf.initialize_all_variables())

            # initialize word embeddings variables in the model
            if self.embedding_chunk_number > 0:
                sess.run(self.model.assign1, feed_dict={self.word_embedding_tf_placeholder: self.word_embedding_matrices[0]})
            if self.embedding_chunk_number > 1:
                sess.run(self.model.assign2, feed_dict={self.word_embedding_tf_placeholder: self.word_embedding_matrices[1]})
            if self.embedding_chunk_number > 2:
                sess.run(self.model.assign3, feed_dict={self.word_embedding_tf_placeholder: self.word_embedding_matrices[2]})
            if self.embedding_chunk_number > 3:
                sess.run(self.model.assign4, feed_dict={self.word_embedding_tf_placeholder: self.word_embedding_matrices[3]})
            if self.embedding_chunk_number > 4:
                sess.run(self.model.assign5, feed_dict={self.word_embedding_tf_placeholder: self.word_embedding_matrices[4]})
            if self.embedding_chunk_number > 5:
                sess.run(self.model.assign6, feed_dict={self.word_embedding_tf_placeholder: self.word_embedding_matrices[5]})
            if self.embedding_chunk_number > 6:
                sess.run(self.model.assign7, feed_dict={self.word_embedding_tf_placeholder: self.word_embedding_matrices[6]})
            if self.embedding_chunk_number > 7:
                sess.run(self.model.assign8, feed_dict={self.word_embedding_tf_placeholder: self.word_embedding_matrices[7]})
            if self.embedding_chunk_number > 8:
                sess.run(self.model.assign9, feed_dict={self.word_embedding_tf_placeholder: self.word_embedding_matrices[8]})

            # initialize embeddings variables in the model
            if self.position_embedding_flag:
                sess.run(self.model.assign_chemical_position_embeddings,
                         feed_dict={self.position_embedding_tf_placeholder: self.position_embedding_matrix})
                sess.run(self.model.assign_protein_position_embeddings,
                         feed_dict={self.position_embedding_tf_placeholder: self.position_embedding_matrix})

            if self.pos_tag_embedding_flag:
                sess.run(self.model.assign_pos_tag_embeddings,
                         feed_dict={self.pos_tag_embedding_tf_placeholder: self.pos_tag_embedding_matrix})

            if self.iob_tag_embedding_flag:
                sess.run(self.model.assign_iob_tag_embeddings,
                         feed_dict={self.iob_tag_embedding_tf_placeholder: self.iob_tag_embedding_matrix})

            for epoch in range(self.num_epoch):
                print("Epoch Number: {}".format(epoch))
                self.report_file.write("\nEpoch Number: {}\n".format(epoch))
                # Training set optimize
                self.optimize_model(tf_sess=sess)

                # training set predict
                training_results = self.run_model(tf_sess=sess,
                                                  dataset_type='training',
                                                  num_step=self.training_num_batch)

                training_epoch_metrics = training_results[0]
                training_epoch_measures = training_results[1]
                training_epoch_predictions = training_results[2]
                training_epoch_logits = training_results[3]
                training_epoch_truth_values = training_results[4]

                print("Training Set Error: Precision:{}, Recall:{}, f1-measure:{}".format(training_epoch_metrics[0],
                                                                                          training_epoch_metrics[1],
                                                                                          training_epoch_metrics[2]))

                self.report_file.write("Training Set Error: Precision:{}, Recall:{}, f1-measure:{}\n".format(training_epoch_metrics[0],
                                                                                                             training_epoch_metrics[1],
                                                                                                             training_epoch_metrics[2]))

                # development set predict
                if self.development_set_flag:
                    development_results = self.run_model(tf_sess=sess,
                                                         dataset_type='development',
                                                         num_step=self.development_num_batch)

                    development_epoch_metrics = development_results[0]
                    development_epoch_measures = development_results[1]
                    development_epoch_predictions = development_results[2]
                    development_epoch_logits = development_results[3]
                    development_epoch_truth_values = development_results[4]

                    print("Development Set Error: Precision:{}, Recall:{}, f1-measure:{}".format(development_epoch_metrics[0],
                                                                                                 development_epoch_metrics[1],
                                                                                                 development_epoch_metrics[2]))

                    self.report_file.write("Development Set Error: Precision:{}, Recall:{}, f1-measure:{}\n".format(development_epoch_metrics[0],
                                                                                                                    development_epoch_metrics[1],
                                                                                                                    development_epoch_metrics[2]))
                # test set predict
                if self.test_set_flag:
                    test_results = self.run_model(tf_sess=sess,
                                                  dataset_type='test',
                                                  num_step=self.test_num_batch)

                    test_epoch_metrics = test_results[0]
                    test_epoch_measures = test_results[1]
                    test_epoch_predictions = test_results[2]
                    test_epoch_logits = test_results[3]
                    test_epoch_truth_values = test_results[4]

                    print("Test Set Error: Precision:{}, Recall:{}, f1-measure:{}".format(test_epoch_metrics[0],
                                                                                          test_epoch_metrics[1],
                                                                                          test_epoch_metrics[2]))

                    self.report_file.write("Test Set Error: Precision:{}, Recall:{}, f1-measure:{}\n".format(test_epoch_metrics[0],
                                                                                                             test_epoch_metrics[1],
                                                                                                             test_epoch_metrics[2]))

                # set best development results and test results
                if self.early_stopping_set == 'development':
                    current_metric = self.development_epoch_metrics[2]
                    best_metric = self.best_development_metrics[2]
                if self.early_stopping_set == 'test':
                    current_metric = self.test_epoch_metrics[2]
                    best_metric = self.best_test_metrics[2]

                if best_metric < current_metric:
                    # set best epoch
                    self.best_epoch = epoch

                    # set best development results
                    if self.early_stopping_set == 'development':
                        self.best_development_metrics[0] = development_epoch_metrics[0]
                        self.best_development_metrics[1] = development_epoch_metrics[1]
                        self.best_development_metrics[2] = development_epoch_metrics[2]
                        self.best_development_truth_values = development_epoch_truth_values
                        self.best_development_predictions = development_epoch_predictions
                        self.best_development_logits = development_epoch_logits

                    # set best test results
                    if self.early_stopping_set == 'test':
                        self.best_test_metrics[0] = test_epoch_metrics[0]
                        self.best_test_metrics[1] = test_epoch_metrics[1]
                        self.best_test_metrics[2] = test_epoch_metrics[2]
                        self.best_test_truth_values = test_epoch_truth_values
                        self.best_test_predictions = test_epoch_predictions
                        self.best_test_logits = test_epoch_logits

                # check early stopping
                if self.best_epoch*2 < epoch and epoch > min_epoch_number:
                    if self.development_set_flag:
                        self.report_file.write("\nBest Development Results:\n")
                        self.report_file.write("Development Precision: {}".format(self.best_development_metrics[0]))
                        self.report_file.write("Development Recall: {}".format(self.best_development_metrics[1]))
                        self.report_file.write("Development F1-measure: {}".format(self.best_development_metrics[2]))

                        pickle.dump(self.best_development_truth_values, open(self.report_directory + '/' +
                                                                             "best_development_truth_values.pkl", "wb"))
                        pickle.dump(self.best_development_predictions, open(self.report_directory + '/' +
                                                                            "best_development_predictions.pkl", "wb"))
                        pickle.dump(self.best_development_logits, open(self.report_directory + '/' +
                                                                       "best_development_logits.pkl", "wb"))
                    if self.test_set_flag:
                        self.report_file.write("\nBest Test Results:\n")
                        self.report_file.write("Test Precision: {}".format(self.best_test_metrics[0]))
                        self.report_file.write("Test Recall: {}".format(self.best_test_metrics[1]))
                        self.report_file.write("Test F1-measure: {}".format(self.best_test_metrics[2]))

                        pickle.dump(self.best_test_truth_values, open(self.report_directory + '/' +
                                                                      "best_test_truth_values.pkl", "wb"))
                        pickle.dump(self.best_test_predictions, open(self.report_directory + '/' +
                                                                     "best_test_predictions.pkl", "wb"))
                        pickle.dump(self.best_test_logits, open(self.report_directory + '/' +
                                                                "best_test_logits.pkl", "wb"))

                    break

        if self.development_set_flag:
            self.report_file.write("\nBest Development Results:\n")
            self.report_file.write("Development Precision: {}".format(self.best_development_metrics[0]))
            self.report_file.write("Development Recall: {}".format(self.best_development_metrics[1]))
            self.report_file.write("Development F1-measure: {}".format(self.best_development_metrics[2]))

            pickle.dump(self.best_development_truth_values, open(self.report_directory + '/' +
                                                                 "best_development_truth_values.pkl", "wb"))
            pickle.dump(self.best_development_predictions, open(self.report_directory + '/' +
                                                                "best_development_predictions.pkl", "wb"))
            pickle.dump(self.best_development_logits, open(self.report_directory + '/' +
                                                           "best_development_logits.pkl", "wb"))
        if self.test_set_flag:
            self.report_file.write("\nBest Test Results:\n")
            self.report_file.write("Test Precision: {}".format(self.best_test_metrics[0]))
            self.report_file.write("Test Recall: {}".format(self.best_test_metrics[1]))
            self.report_file.write("Test F1-measure: {}".format(self.best_test_metrics[2]))

            pickle.dump(self.best_test_truth_values, open(self.report_directory + '/' +
                                                          "best_test_truth_values.pkl", "wb"))
            pickle.dump(self.best_test_predictions, open(self.report_directory + '/' +
                                                         "best_test_predictions.pkl", "wb"))
            pickle.dump(self.best_test_logits, open(self.report_directory + '/' +
                                                    "best_test_logits.pkl", "wb"))

    def get_development_results(self):
        results = [self.best_development_metrics,
                   self.best_development_predictions,
                   self.best_development_logits,
                   self.best_development_truth_values]

        return results

    def get_test_results(self):
        results = [self.best_test_metrics,
                   self.best_test_predictions,
                   self.best_test_logits,
                   self.best_test_truth_values]
        return results

    def get_best_epoch(self):
        return self.best_epoch

    @staticmethod
    def get_metrics(logits, labels):
        false_negative = []
        false_positive = []

        logits = np.argmax(logits, axis=1)
        labels = np.asarray(labels, dtype=np.float32)
        labels = np.argmax(labels, axis=1)
        positive = np.sum(labels)

        batch_result = np.subtract(labels, logits)
        for i in range(len(batch_result)):
            res = batch_result[i]
            if res == 1:
                false_negative.append(i)
            elif res == -1:
                false_positive.append(i)

        return false_negative, false_positive, positive

    @staticmethod
    def calculate_metrics(false_negative, false_positive, positive_labels):
        true_positive = positive_labels - false_negative
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1_measure = 2 * (recall * precision) / (recall + precision)
        return precision, recall, f1_measure

    def optimize_model(self, tf_sess):

        # configure progress bar
        progress_bar = progressbar.ProgressBar(maxval=self.training_num_batch,
                                               widgets=["Training Set Optimize: ",
                                                        progressbar.Bar('=', '[', ']'),
                                                        ' ',
                                                        progressbar.Percentage()])
        progress_bar_counter = 0
        progress_bar.start()

        for batch in range(self.training_num_batch):

            # obtain batch data
            batch_data = self.data_interface.get_batch(dataset_type='training')

            # fetch batch data
            batch_word_ids = batch_data[0]
            batch_seq_lens = batch_data[1]
            batch_labels = batch_data[2]
            if self.pos_tag_embedding_flag:
                batch_pos_ids = batch_data[3]
            if self.iob_tag_embedding_flag:
                batch_iob_ids = batch_data[4]
            if self.position_embedding_flag:
                batch_distance_protein = batch_data[5]
                batch_distance_chemical = batch_data[6]

            if len(batch_labels) == self.data_interface.batch_size:
                batch_feed_dict = {self.word_ids_tf_placeholder: batch_word_ids,
                                   self.label_tf_placeholder: batch_labels,
                                   self.seq_len_tf_placeholder: batch_seq_lens}

                if self.pos_tag_embedding_flag:
                    batch_feed_dict[self.pos_tag_tf_placeholder] = batch_pos_ids
                if self.iob_tag_embedding_flag:
                    batch_feed_dict[self.iob_tag_tf_placeholder] = batch_iob_ids
                if self.position_embedding_flag:
                    batch_feed_dict[self.distance_protein_tf_placeholder] = batch_distance_protein
                    batch_feed_dict[self.distance_chemical_tf_placeholder] = batch_distance_chemical

                tf_sess.run(self.model.optimize, feed_dict=batch_feed_dict)

            progress_bar_counter = progress_bar_counter + 1
            progress_bar.update(progress_bar_counter)

        progress_bar.finish()

    def run_model(self, tf_sess, dataset_type, num_step):
        metrics = [0, 0, 0]  # precision, recall, f1-measure
        measures = [0, 0, 0]  # fn, fp, positive_labels
        predictions = []
        logits = []
        truth_values = []

        progress_bar = progressbar.ProgressBar(maxval=num_step,
                                               widgets=["Evaluation ({}) ".format(dataset_type),
                                                        progressbar.Bar('=', '[', ']'),
                                                        ' ',
                                                        progressbar.Percentage()])
        progress_bar_counter = 0
        progress_bar.start()

        for step in range(num_step):
            # obtain batch data
            batch_data = self.data_interface.get_batch(dataset_type=dataset_type)

            # fetch batch data
            batch_word_ids = batch_data[0]
            batch_seq_lens = batch_data[1]
            batch_labels = batch_data[2]
            if self.pos_tag_embedding_flag:
                batch_pos_ids = batch_data[3]
            if self.iob_tag_embedding_flag:
                batch_iob_ids = batch_data[4]
            if self.position_embedding_flag:
                batch_distance_protein = batch_data[5]
                batch_distance_chemical = batch_data[6]

            if len(batch_labels) == self.data_interface.batch_size:
                batch_feed_dict = {self.word_ids_tf_placeholder: batch_word_ids,
                                   self.label_tf_placeholder: batch_labels,
                                   self.seq_len_tf_placeholder: batch_seq_lens}

                if self.pos_tag_embedding_flag:
                    batch_feed_dict[self.pos_tag_tf_placeholder] = batch_pos_ids
                if self.iob_tag_embedding_flag:
                    batch_feed_dict[self.iob_tag_tf_placeholder] = batch_iob_ids
                if self.position_embedding_flag:
                    batch_feed_dict[self.distance_protein_tf_placeholder] = batch_distance_protein
                    batch_feed_dict[self.distance_chemical_tf_placeholder] = batch_distance_chemical

                batch_logits = tf_sess.run(self.model.prediction, feed_dict=batch_feed_dict)

                for logit in batch_logits:
                    logits.append(logit)

                batch_predictions = np.argmax(batch_logits, axis=1)
                for prediction in batch_predictions:
                    predictions.append(prediction)

                batch_truth_values = np.argmax(batch_labels, axis=1)
                for truth_value in batch_truth_values:
                    truth_values.append(truth_value)

                batch_fn, batch_fp, batch_positive = self.get_metrics(batch_logits, batch_labels)
                measures[0] += len(batch_fn)
                measures[1] += len(batch_fp)
                measures[2] += batch_positive

            progress_bar_counter = progress_bar_counter + 1
            progress_bar.update(progress_bar_counter)

        progress_bar.finish()

        metrics[0], metrics[1], metrics[2] = self.calculate_metrics(measures[0], measures[1], measures[2])
        return [metrics, measures, predictions, logits, truth_values]
