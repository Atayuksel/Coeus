import numpy as np
import tensorflow as tf
import progressbar
import data_interface as di
import bilstm_model as bilstm

data_interface = di.DataInterface(dataset_name='BioCreative',
                                  embedding_dir='dataset/glove.6B/glove.6B.50d.txt')

max_seq_length = data_interface.dataset['training']['max_seq_len']
embedding_matrix = data_interface.embeddings
embedding_dimension = embedding_matrix.shape[1]
vocabulary_size = embedding_matrix.shape[0]
batch_size = 10
num_epoch = 2
num_batch_in_epoch = len(data_interface.dataset['training']['data']) + 1

# tensorflow placeholder
data_ph = tf.placeholder(tf.int64, [batch_size, max_seq_length], name='data_placeholder')
labels_ph = tf.placeholder(tf.float32, [batch_size, 2], name='label_placeholder')
embedding_ph = tf.placeholder(tf.float32, [vocabulary_size, embedding_dimension], name='embedding_placeholder')
seq_lens_ph = tf.placeholder(tf.int64, [batch_size, ], name='sequence_length_placeholder')

# network hyper-parameters
class_weights = tf.constant([[1., 1.]])
num_hidden = 2
learning_rate = 0.01

# tensorflow model
model = bilstm.BiLSTMModel(data=data_ph,
                           target=labels_ph,
                           seq_lens=seq_lens_ph,
                           class_weights=class_weights,
                           num_hidden=num_hidden,
                           learning_rate=learning_rate,
                           embedding_size=embedding_dimension,
                           vocab_size=vocabulary_size)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    word_embedding_init = model.embedding_v.assign(embedding_ph)
    sess.run(word_embedding_init, feed_dict={embedding_ph: embedding_matrix})

    progress_bar = progressbar.ProgressBar(maxval=num_batch_in_epoch,
                                           widgets=[progressbar.Bar('=', '[', ']'),
                                                    ' ',
                                                    progressbar.Percentage()])
    for epoch in range(num_epoch):
        progress_bar_counter = 0
        progress_bar.start()
        for batch in range(num_batch_in_epoch):
            batch_data, batch_labels, batch_seq_lens = data_interface.get_batch(dataset_type='training')
            if len(batch_labels) == batch_size:
                sess.run(model.optimize, feed_dict={data_ph: batch_data,
                                                    labels_ph: batch_labels,
                                                    seq_lens_ph: batch_seq_lens})

            progress_bar_counter = progress_bar_counter + 1
            progress_bar.update(progress_bar_counter)
        progress_bar.finish()

        # testing
        fp = 0
        fn = 0
        positive_labels = 0
        for batch in range(num_batch_in_epoch):
            batch_data, batch_labels, batch_seq_lens = data_interface.get_batch(dataset_type='training')
            if len(batch_labels) == batch_size:
                batch_prediction = sess.run(model.prediction, feed_dict={data_ph: batch_data,
                                                                         labels_ph: batch_labels,
                                                                         seq_lens_ph: batch_seq_lens})

                # batch_prediction = sess.run(tf.argmax(batch_prediction, 1))
                batch_prediction = np.argmax(batch_prediction, axis=1)
                batch_labels = np.asarray(batch_labels, dtype=np.float32)
                batch_labels = np.argmax(batch_labels, axis=1)
                # batch_labels = sess.run(tf.argmax(batch_labels, 1))
                positive_labels = positive_labels + np.sum(batch_labels)

                batch_result = np.subtract(batch_labels, batch_prediction)
                unique, counts = np.unique(batch_result, return_counts=True)
                freqs = dict(zip(unique, counts))
                for key, value in freqs.items():
                    if key == 1:
                        fn = fn + value
                    elif key == -1:
                        fp = fp + value

        tp = positive_labels-fn
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_measure = 2 * (recall * precision) / (recall + precision)

        print("Epoch Number: {}, Precision:{}, Recall:{}, f1-measure:{}".format(epoch, precision, recall, f1_measure))