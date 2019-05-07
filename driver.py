import os
import math
import datetime
import numpy as np
import tensorflow as tf
import progressbar
import data_interface as di
import bilstm_model as bilstm


def get_metrics(logits, labels):
    false_negative = 0
    false_positive = 0

    logits = np.argmax(logits, axis=1)
    labels = np.asarray(labels, dtype=np.float32)
    labels = np.argmax(labels, axis=1)
    positive = np.sum(labels)

    batch_result = np.subtract(labels, logits)
    unique, counts = np.unique(batch_result, return_counts=True)
    freqs = dict(zip(unique, counts))
    for key, value in freqs.items():
        if key == 1:
            false_negative = false_negative + value
        elif key == -1:
            false_positive = false_positive + value

    return false_negative, false_positive, positive


def calculate_metrics(false_negative, false_positive, positive_labels):
    true_positive = positive_labels - false_negative
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_measure = 2 * (recall * precision) / (recall + precision)
    return precision, recall, f1_measure


# network hyper-parameters
network_type = "BiLSTM"
batch_size = 10
num_epoch = 10
class_weights = tf.constant([[1., 1.]])
num_hidden = 1024
learning_rate = 0.01

# data
data_interface = di.DataInterface(dataset_name='BioCreative',
                                  embedding_dir='dataset/glove.6B/glove.6B.50d.txt',
                                  batch_size=batch_size)
max_seq_length = data_interface.dataset['training']['max_seq_len']
embedding_matrix = data_interface.embeddings
embedding_dimension = embedding_matrix.shape[1]
vocabulary_size = embedding_matrix.shape[0]

# training data
tra_num_batch_in_epoch = math.ceil(len(data_interface.dataset['training']['data']) / batch_size)

# development data
dev_num_batch_in_epoch = math.ceil(len(data_interface.dataset['development']['data']) / batch_size)

# evaluation metrics
positive_labels = 0
fp = 0
fn = 0
precision = 0
recall = 0
f1_measure = 0

# tensorflow placeholder
data_ph = tf.placeholder(tf.int64, [batch_size, max_seq_length], name='data_placeholder')
labels_ph = tf.placeholder(tf.float32, [batch_size, 2], name='label_placeholder')
embedding_ph = tf.placeholder(tf.float32, [vocabulary_size, embedding_dimension], name='embedding_placeholder')
seq_lens_ph = tf.placeholder(tf.int64, [batch_size, ], name='sequence_length_placeholder')

# tensorflow model
model = bilstm.BiLSTMModel(data=data_ph,
                           target=labels_ph,
                           seq_lens=seq_lens_ph,
                           class_weights=class_weights,
                           num_hidden=num_hidden,
                           learning_rate=learning_rate,
                           embedding_size=embedding_dimension,
                           vocab_size=vocabulary_size)

# prepare report file
currentDT = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
report_name = "report_" + str(currentDT) + ".txt"
report_file = open(report_name, "w+")
line = "Time: " + str(datetime.datetime.now()) + "\n"
report_file.write(line)
line = "Network: " + network_type + "\n"
report_file.write(line)
line = "Batch size: " + str(batch_size) + "\n"
report_file.write(line)
line = "Number of Epochs: " + str(num_epoch) + "\n"
report_file.write(line)
line = "Number of Hidden Layers: " + str(num_hidden) + "\n"
report_file.write(line)
line = "Learning Rate: " + str(learning_rate) + "\n"
report_file.write(line)
line = "\nTraining Error \n"
report_file.write(line)

# create a session and run the graph
with tf.Session() as sess:
    # initialize variables
    sess.run(tf.initialize_all_variables())
    word_embedding_init = model.embedding_v.assign(embedding_ph)
    sess.run(word_embedding_init, feed_dict={embedding_ph: embedding_matrix})

    for epoch in range(num_epoch):
        # training set optimize
        progress_bar = progressbar.ProgressBar(maxval=tra_num_batch_in_epoch,
                                               widgets=["Epoch:{} (Optimize) ".format(epoch),
                                                        progressbar.Bar('=', '[', ']'),
                                                        ' ',
                                                        progressbar.Percentage()])
        progress_bar_counter = 0
        progress_bar.start()

        for batch in range(tra_num_batch_in_epoch):
            batch_data, batch_labels, batch_seq_lens = data_interface.get_batch(dataset_type='training')
            if len(batch_labels) == batch_size:
                sess.run(model.optimize, feed_dict={data_ph: batch_data,
                                                    labels_ph: batch_labels,
                                                    seq_lens_ph: batch_seq_lens})

            progress_bar_counter = progress_bar_counter + 1
            progress_bar.update(progress_bar_counter)
        progress_bar.finish()

        # training set predict
        progress_bar = progressbar.ProgressBar(maxval=tra_num_batch_in_epoch,
                                               widgets=["Epoch:{} (Predict) ".format(epoch),
                                                        progressbar.Bar('=', '[', ']'),
                                                        ' ',
                                                        progressbar.Percentage()])
        progress_bar_counter = 0
        progress_bar.start()
        for batch in range(tra_num_batch_in_epoch):
            batch_data, batch_labels, batch_seq_lens = data_interface.get_batch(dataset_type='training')
            if len(batch_labels) == batch_size:
                batch_prediction = sess.run(model.prediction, feed_dict={data_ph: batch_data,
                                                                         labels_ph: batch_labels,
                                                                         seq_lens_ph: batch_seq_lens})

                batch_fn, batch_fp, batch_positive = get_metrics(batch_prediction, batch_labels)
                fn += batch_fn
                fp += batch_fp
                positive_labels += batch_positive

            progress_bar_counter = progress_bar_counter + 1
            progress_bar.update(progress_bar_counter)
        progress_bar.finish()
        precision, recall, f1_measure = calculate_metrics(fn, fp, positive_labels)

        print("Epoch Number: {}, Precision:{}, Recall:{}, f1-measure:{}\n".format(epoch, precision, recall, f1_measure))
        report_file.write(
            "Epoch Number: {}, Precision:{}, Recall:{}, f1-measure:{}\n".format(epoch, precision, recall, f1_measure))

    # development set evaluation
    positive_labels = 0
    fp = 0
    fn = 0
    precision = 0
    recall = 0
    f1_measure = 0
    progress_bar = progressbar.ProgressBar(maxval=dev_num_batch_in_epoch,
                                           widgets=["Development Set Test",
                                                    progressbar.Bar('=', '[', ']'),
                                                    ' ',
                                                    progressbar.Percentage()])
    progress_bar_counter = 0
    progress_bar.start()
    for batch in range(dev_num_batch_in_epoch):
        batch_data, batch_labels, batch_seq_lens = data_interface.get_batch(dataset_type='development')
        if len(batch_labels) == batch_size:
            batch_prediction = sess.run(model.prediction, feed_dict={data_ph: batch_data,
                                                                     labels_ph: batch_labels,
                                                                     seq_lens_ph: batch_seq_lens})
            batch_fn, batch_fp, batch_positive = get_metrics(batch_prediction, batch_labels)
            fn += batch_fn
            fp += batch_fp
            positive_labels += batch_positive
        progress_bar_counter = progress_bar_counter + 1
        progress_bar.update(progress_bar_counter)
    progress_bar.finish()
    precision, recall, f1_measure = calculate_metrics(fn, fp, positive_labels)
    print("Development Set Evaluation: Precision:{}, Recall:{}, f1-measure:{}\n".format(precision, recall, f1_measure))
    report_file.write(
        "\nDevelopment Error \nPrecision:{}, Recall:{}, f1-measure:{} \n".format(precision, recall, f1_measure))

report_file.close()