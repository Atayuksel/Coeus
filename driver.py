import os
import math
import datetime
import numpy as np
import tensorflow as tf
import progressbar
import data_interface as di
import bilstm_model as bilstm
import paper_models as papers
import configparser
import sys


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


def calculate_metrics(false_negative, false_positive, positive_labels):
    true_positive = positive_labels - false_negative
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_measure = 2 * (recall * precision) / (recall + precision)
    return precision, recall, f1_measure


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# open configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# create result file based on the run_type
run_type = config['BASE']['run_type']  # single, grid_search
report_name = "none"
file_open_type = "none"
if run_type == 'grid_search':
    report_name = config['BASE']['gridsearch_report_file_name']
    file_open_type = "a+"
elif run_type == 'single':
    currentDT = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    report_name = "report_" + str(currentDT) + ".txt"
    file_open_type = "w+"

if report_name != "none" and file_open_type != "none":
    report_file = open(report_name, file_open_type)
else:
    print("Report File Opening Error")
    sys.exit()

# BASE TYPE PARAMETERS
# current time
line = "Time: {}\n".format(str(datetime.datetime.now()))
print(line)
report_file.write(line)

# run type
line = "Run Type: {}\n".format(run_type)
print(line)
report_file.write(line)

# model type
model_type = config['BASE']['model_type']  # BILSTM, CNN1, CNN2, CNN3, KimCNN
line = "Model Type: {}\n".format(model_type)
print(line)
report_file.write(line)

# batch size
batch_size = int(config['BASE']['batch_size'])  # batch size for the data interface
line = "Batch Size: {}\n".format(str(batch_size))
print(line)
report_file.write(line)

# epoch size
num_epoch = int(config['BASE']['num_epoch'])  # epoch number for training
line = "Number of Epoch: {}\n".format(str(num_epoch))
print(line)
report_file.write(line)

# learning rate
learning_rate = float(config['BASE']['learning_rate'])
line = "Learning Rate: {}\n".format(str(learning_rate))
print(line)
report_file.write(line)

# text selection
text_selection = config['BASE']['text_selection']
line = "Text Selection: {}\n".format(text_selection)
print(line)
report_file.write(line)

# Relation Type: binary
relation_type = config['BASE']['relation_type']
line = "Relation Type: {}\n".format(relation_type)
print(line)
report_file.write(line)

# Error Weights
error_weight = config['BASE']['error_weight']
if error_weight == 'unweighted':
    line = "Error Function: {}\n".format('Unweighted Cross Entropy')
    print(line)
    report_file.write(line)
elif error_weight == 'weighted':
    line = "Error Function: {}\n".format('Weighted Cross Entropy')
    print(line)
    report_file.write(line)
    class_weights = tf.constant([[1., 1.]])

# EMBEDDINGS
section_name = 'EMBEDDINGS'

# word embeddings
word_embedding_type = config[section_name]['word_embedding_type']  # selected word embedding for training.
line = "Word Embedding Type: {}\n".format(word_embedding_type)
print(line)
report_file.write(line)

word_embedding_size = config[section_name]['word_embedding_size']  # selected embedding size for the word embeddings.
line = "Word Embedding Size: {}\n".format(str(word_embedding_size))
print(line)
report_file.write(line)

# position embeddings
position_embedding_flag = config[section_name]['position_embedding_flag']
line = "Posistion Embeddings Flag: {}\n".format(position_embedding_flag)
print(line)
report_file.write(line)
position_embedding_flag = True if position_embedding_flag == 'true' else False
position_embedding_size = 0
if position_embedding_flag:
    position_embedding_size = int(config[section_name]['position_embedding_size'])
    line = "Position Embeddings Size: {}\n".format(str(position_embedding_size))
    print(line)
    report_file.write(line)

# Pos Tag Embeddings
pos_tag_embedding_flag = config[section_name]['pos_tag_embedding_flag']
line = "Pos Tag Embedding Flag: {}\n".format(pos_tag_embedding_flag)
print(line)
report_file.write(line)
pos_tag_embedding_flag = True if pos_tag_embedding_flag == 'true' else False
pos_tag_embedding_size = 0
if pos_tag_embedding_flag:
    pos_tag_embedding_size = int(config[section_name]['pos_tag_embedding_size'])
    line = "Pos Tag Embeddings Size: {}\n".format(str(pos_tag_embedding_size))
    print(line)
    report_file.write(line)

# IOB Chunk Tags Embeddings
iob_embedding_flag = config[section_name]['iob_embedding_flag']
line = "IOB Embedding Flag: {}\n".format(iob_embedding_flag)
print(line)
report_file.write(line)
iob_embedding_flag = True if iob_embedding_flag == 'true' else False
iob_embedding_size = 0
if iob_embedding_flag:
    iob_embedding_size = int(config[section_name]['iob_embedding_size'])
    line = "IOB embedding size: {}\n".format(str(iob_embedding_size))
    print(line)
    report_file.write(line)

# DATA INTERFACE
print("Setting Up Data Interface")

# word embedding file
pre_embedding_directory = "none"
binary_relation = "none"
if word_embedding_type == 'biomedical':
    pre_embedding_directory = 'dataset/PubMed-shuffle-win-2.txt'
    line = "Embedding Directory: {}\n".format(pre_embedding_directory)
    print(line)
    report_file.write(line)
elif word_embedding_type == 'glove':
    pre_embedding_directory = 'dataset/glove.6B/glove.6B.' + word_embedding_size + 'd.txt'
    line = "Embedding Directory: {}\n".format(pre_embedding_directory)
    print(line)
    report_file.write(line)

# relation type: binary or multiple
if relation_type == 'binary':
    binary_relation = True
    line = "Binary Relation: {}\n".format(str(binary_relation))
    print(line)
    report_file.write(line)
elif relation_type == 'multiple':
    binary_relation = False
    line = "Binary Relation: {}\n".format(str(binary_relation))
    print(line)
    report_file.write(line)

if pre_embedding_directory != "none" and binary_relation != "none":

    # create data interface object
    data_interface = di.DataInterface(dataset_name='BioCreative',
                                      embedding_dir=pre_embedding_directory,
                                      batch_size=batch_size,
                                      text_selection=text_selection,
                                      binary_relation=binary_relation)
    print("Data Interface is created successfully.")

    max_train_seq_length = data_interface.dataset['training']['max_seq_len']
    line = "Maximum Training Sequence Length: {}\n".format(str(max_train_seq_length))
    print(line)
    report_file.write(line)

    # get embedding matrix and divide it to 8.
    embedding_matrix = data_interface.embeddings
    embedding_matrices = np.split(embedding_matrix, 8)
    embedding_dimension = embedding_matrix.shape[1]
    vocabulary_size = embedding_matrix.shape[0]
    max_train_position_distance = len(data_interface.pos_to_id)

    # check embedding sizes
    assert embedding_dimension == int(word_embedding_size)

    line = "Vocabulary Size: {}\n".format(str(vocabulary_size))
    print(line)
    report_file.write(line)

    # training data
    tra_num_batch_in_epoch = math.ceil(len(data_interface.dataset['training']['data']) / batch_size)
    # development data
    dev_num_batch_in_epoch = math.ceil(len(data_interface.dataset['development']['data']) / batch_size)
    # test data
    test_num_batch_in_epoch = math.ceil(len(data_interface.dataset['test']['data']) / batch_size)

else:
    print("Data Interface Error")
    sys.exit()

# TENSORFLOW
print("Create Tensorflow Placeholders")

# tensorflow placeholder
data_ph = tf.placeholder(tf.int64, [batch_size, 144], name='data_placeholder')
seq_lens_ph = tf.placeholder(tf.int64, [batch_size, ], name='sequence_length_placeholder')
protein_distance = tf.placeholder(tf.int64, [batch_size, 144], name='distance_protein_placeholder')
chemical_distance = tf.placeholder(tf.int64, [batch_size, 144], name='distance_chemical_placeholder')
data_pos_tags_ph = tf.placeholder(tf.int64, [batch_size, 144], name='data_pos_tags_placeholder')
data_iob_tags_ph = tf.placeholder(tf.int64, [batch_size, 144], name='data_iob_tags_placeholder')
labels_ph = tf.placeholder(tf.float32, [batch_size, 2], name='label_placeholder')
embedding_ph = tf.placeholder(tf.float32, [embedding_matrices[0].shape[0], embedding_dimension],
                              name='embedding_placeholder')

# tensorflow models
print('Creating Tensorflow Model')
model = "none"

if model_type == "bilstm":
    line = "Selected Model: {}".format(model_type)
    print(line)
    report_file.write(line)

    # obtain bi-lstm specific parameter
    section_name = 'BILSTM'
    lstm_hidden_unit_size = int(config[section_name]['lstm_hidden_unit'])
    line = "LSTM Hidden Unit Size: {}\n".format(lstm_hidden_unit_size)
    print(line)
    report_file.write(line)

    # define the tf model
    model = bilstm.BiLSTMModel(data=data_ph,
                               target=labels_ph,
                               seq_lens=seq_lens_ph,
                               learning_rate=learning_rate,
                               embedding_dimension=embedding_dimension,
                               vocab_size=vocabulary_size,
                               embedding_placeholder=embedding_ph,
                               lstm_hidden_unit_size=lstm_hidden_unit_size,
                               max_distance=max_train_position_distance,
                               position_embedding_size=position_embedding_size,
                               distance_chemical=chemical_distance,
                               distance_protein=protein_distance,
                               data_pos_tags=data_pos_tags_ph,
                               pos_tag_embedding_size=pos_tag_embedding_size,
                               data_iob_tags=data_iob_tags_ph,
                               iob_tag_embedding_size=iob_embedding_size)

    print("{} model is created".format(model_type))

elif model_type == "KimCNN":
    line = "Selected Model: {}".format(model_type)
    print(line)
    report_file.write(line)

    # obtain CNN specific parameters
    section_name = "CNN"
    cnn_filter_out = int(config[section_name]['cnn_filter_out'])
    line = "CNN Filter Out: {}\n".format(cnn_filter_out)
    print(line)
    report_file.write(line)

    cnn_hidden_unit = int(config[section_name]['cnn_hidden_unit'])
    line = "CNN Hidden Unit: {}\n".format(cnn_hidden_unit)
    print(line)
    report_file.write(line)

    model = papers.KimCNN(data=data_ph,
                          target=labels_ph,
                          distance_protein=protein_distance,
                          distance_chemical=chemical_distance,
                          hidden_unit_size=cnn_hidden_unit,
                          embedding_placeholder=embedding_ph,
                          embedding_size=embedding_dimension,
                          vocabulary_size=vocabulary_size,
                          learning_rate=learning_rate,
                          position_embedding_size=position_embedding_size,
                          max_position_distance=max_train_position_distance,
                          filter_size=cnn_filter_out,
                          data_pos_tags=data_pos_tags_ph,
                          pos_tag_embedding_size=pos_tag_embedding_size)

if model == "none":
    print("Error occurred in Model Creation")
    sys.exit()

# evaluation metrics
positive_labels = 0
fp = 0
fn = 0
precision = 0
recall = 0
f1_measure = 0

print("TF Model {} is created successfully\n".format(model_type))

# create a session and run the graph
with tf.Session() as sess:
    print("TF session is created successfully\n")
    # initialize variables
    sess.run(tf.initialize_all_variables())
    sess.run(model.assign1, feed_dict={embedding_ph: embedding_matrices[0]})
    sess.run(model.assign2, feed_dict={embedding_ph: embedding_matrices[1]})
    sess.run(model.assign3, feed_dict={embedding_ph: embedding_matrices[2]})
    sess.run(model.assign4, feed_dict={embedding_ph: embedding_matrices[3]})
    sess.run(model.assign5, feed_dict={embedding_ph: embedding_matrices[4]})
    sess.run(model.assign6, feed_dict={embedding_ph: embedding_matrices[5]})
    sess.run(model.assign7, feed_dict={embedding_ph: embedding_matrices[6]})
    sess.run(model.assign8, feed_dict={embedding_ph: embedding_matrices[7]})

    for epoch in range(num_epoch):
        print("Epoch Number: {}\n".format(str(epoch)))

        # TRAINING SET OPTIMIZE
        progress_bar = progressbar.ProgressBar(maxval=tra_num_batch_in_epoch,
                                               widgets=["Epoch:{} (Training Set Optimize) ".format(epoch),
                                                        progressbar.Bar('=', '[', ']'),
                                                        ' ',
                                                        progressbar.Percentage()])
        progress_bar_counter = 0
        progress_bar.start()
        for batch in range(tra_num_batch_in_epoch):

            batch_data, batch_pos_ids, batch_iob_ids, batch_labels, batch_seq_lens, \
                batch_protein_distance, batch_chemical_distance = data_interface.get_batch(dataset_type='training')

            if len(batch_labels) == batch_size:
                sess.run(model.optimize, feed_dict={data_ph: batch_data,
                                                    data_pos_tags_ph: batch_pos_ids,
                                                    labels_ph: batch_labels,
                                                    seq_lens_ph: batch_seq_lens,
                                                    data_iob_tags_ph: batch_iob_ids,
                                                    protein_distance: batch_protein_distance,
                                                    chemical_distance: batch_chemical_distance})

            progress_bar_counter = progress_bar_counter + 1
            progress_bar.update(progress_bar_counter)
        progress_bar.finish()

        # TRAINING SET PREDICT
        positive_labels = 0
        fp = 0
        fn = 0
        precision = 0
        recall = 0
        f1_measure = 0

        progress_bar = progressbar.ProgressBar(maxval=tra_num_batch_in_epoch,
                                               widgets=["Epoch:{} (Training Set Test) ".format(epoch),
                                                        progressbar.Bar('=', '[', ']'),
                                                        ' ',
                                                        progressbar.Percentage()])
        progress_bar_counter = 0
        progress_bar.start()
        for batch in range(tra_num_batch_in_epoch):
            batch_data, batch_pos_ids, batch_iob_ids, batch_labels, batch_seq_lens, \
                batch_protein_distance, batch_chemical_distance = data_interface.get_batch(dataset_type='training')

            if len(batch_labels) == batch_size:
                batch_prediction = sess.run(model.prediction, feed_dict={data_ph: batch_data,
                                                                         data_pos_tags_ph: batch_pos_ids,
                                                                         labels_ph: batch_labels,
                                                                         seq_lens_ph: batch_seq_lens,
                                                                         data_iob_tags_ph: batch_iob_ids,
                                                                         protein_distance: batch_protein_distance,
                                                                         chemical_distance: batch_chemical_distance})

                batch_fn, batch_fp, batch_positive = get_metrics(batch_prediction, batch_labels)
                fn += len(batch_fn)
                fp += len(batch_fp)
                positive_labels += batch_positive
            progress_bar_counter = progress_bar_counter + 1
            progress_bar.update(progress_bar_counter)
        progress_bar.finish()
        precision, recall, f1_measure = calculate_metrics(fn, fp, positive_labels)
        if run_type == "single":
            print("Training Set Error: {}, Precision:{}, Recall:{}, f1-measure:{}\n".format(epoch, precision, recall,
                                                                                            f1_measure))
            report_file.write("Training Set Error: {}, Precision:{}, Recall:{}, f1-measure:{}\n".format(epoch,
                                                                                                        precision,
                                                                                                        recall,
                                                                                                        f1_measure))
        if run_type == "grid_search":
            print("Training Set Error: {}, Precision:{}, Recall:{}, f1-measure:{}\n".format(epoch, precision, recall,
                                                                                            f1_measure))
            report_file.write("Training Set Error: {}, Precision:{}, Recall:{}, f1-measure:{}\n".format(epoch,
                                                                                                        precision,
                                                                                                        recall,
                                                                                                        f1_measure))

        # DEVELOPMENT SET PREDICT
        positive_labels = 0
        fp = 0
        fn = 0
        precision = 0
        recall = 0
        f1_measure = 0

        progress_bar = progressbar.ProgressBar(maxval=dev_num_batch_in_epoch,
                                               widgets=["Epoch:{} (Development Set Test)".format(epoch),
                                                        progressbar.Bar('=', '[', ']'),
                                                        ' ',
                                                        progressbar.Percentage()])
        progress_bar_counter = 0
        progress_bar.start()
        for batch in range(dev_num_batch_in_epoch):

            batch_data, batch_pos_ids, batch_iob_ids, batch_labels, batch_seq_lens, \
                batch_protein_distance, batch_chemical_distance = data_interface.get_batch(dataset_type='development')

            if len(batch_labels) == batch_size:
                batch_prediction = sess.run(model.prediction, feed_dict={data_ph: batch_data,
                                                                         data_pos_tags_ph: batch_pos_ids,
                                                                         labels_ph: batch_labels,
                                                                         seq_lens_ph: batch_seq_lens,
                                                                         data_iob_tags_ph: batch_iob_ids,
                                                                         protein_distance: batch_protein_distance,
                                                                         chemical_distance: batch_chemical_distance})

                batch_fn, batch_fp, batch_positive = get_metrics(batch_prediction, batch_labels)
                fn += len(batch_fn)
                fp += len(batch_fp)
                positive_labels += batch_positive
                data_interface.add_false_negative('development', batch_fn)
                data_interface.add_false_positive('development', batch_fp)
            progress_bar_counter = progress_bar_counter + 1
            progress_bar.update(progress_bar_counter)
        progress_bar.finish()
        precision, recall, f1_measure = calculate_metrics(fn, fp, positive_labels)

        if run_type == "single":
            print("Development Set Evaluation: Precision:{}, Recall:{}, f1-measure:{}\n".format(precision, recall,
                                                                                                f1_measure))
            report_file.write("\nDevelopment Error \nPrecision:{}, Recall:{}, f1-measure:{} \n".format(precision,
                                                                                                       recall,
                                                                                                       f1_measure))
            data_interface.write_results('development')
        elif run_type == 'grid_search':
            print("Development Set Evaluation: Precision:{}, Recall:{}, f1-measure:{}\n".format(precision, recall,
                                                                                                f1_measure))
            report_file.write("Development Error \nPrecision:{}, Recall:{}, f1-measure:{} \n".format(precision,
                                                                                                     recall,
                                                                                                     f1_measure))

        # TEST SET PREDICTION
        positive_labels = 0
        fp = 0
        fn = 0
        precision = 0
        recall = 0
        f1_measure = 0

        progress_bar = progressbar.ProgressBar(maxval=test_num_batch_in_epoch,
                                               widgets=["Epoch:{} (Development Set Test)".format(epoch),
                                                        progressbar.Bar('=', '[', ']'),
                                                        ' ',
                                                        progressbar.Percentage()])
        progress_bar_counter = 0
        progress_bar.start()
        for batch in range(test_num_batch_in_epoch):

            batch_data, batch_pos_ids, batch_iob_ids, batch_labels, batch_seq_lens, \
                batch_protein_distance, batch_chemical_distance = data_interface.get_batch(dataset_type='test')

            if len(batch_labels) == batch_size:
                batch_prediction = sess.run(model.prediction, feed_dict={data_ph: batch_data,
                                                                         data_pos_tags_ph: batch_pos_ids,
                                                                         labels_ph: batch_labels,
                                                                         seq_lens_ph: batch_seq_lens,
                                                                         data_iob_tags_ph: batch_iob_ids,
                                                                         protein_distance: batch_protein_distance,
                                                                         chemical_distance: batch_chemical_distance})

                batch_fn, batch_fp, batch_positive = get_metrics(batch_prediction, batch_labels)
                fn += len(batch_fn)
                fp += len(batch_fp)
                positive_labels += batch_positive
                data_interface.add_false_negative('test', batch_fn)
                data_interface.add_false_positive('test', batch_fp)
            progress_bar_counter = progress_bar_counter + 1
            progress_bar.update(progress_bar_counter)
        progress_bar.finish()
        precision, recall, f1_measure = calculate_metrics(fn, fp, positive_labels)

        if run_type == "single":
            print("Test Set Evaluation: Precision:{}, Recall:{}, f1-measure:{}\n".format(precision, recall,
                                                                                         f1_measure))
            report_file.write("\nDevelopment Error \nPrecision:{}, Recall:{}, f1-measure:{} \n".format(precision,
                                                                                                       recall,
                                                                                                       f1_measure))
            data_interface.write_results('development')
        elif run_type == 'grid_search':
            print("Test Set Evaluation: Precision:{}, Recall:{}, f1-measure:{}\n".format(precision, recall,
                                                                                         f1_measure))
            report_file.write("Development Error \nPrecision:{}, Recall:{}, f1-measure:{} \n".format(precision,
                                                                                                     recall,
                                                                                                     f1_measure))

report_file.close()
