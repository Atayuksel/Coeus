import numpy as np
import tensorflow as tf

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
data_ph = tf.placeholder(tf.int64, [batch_size, max_seq_length])
labels_ph = tf.placeholder(tf.float32, [batch_size, 2])
embedding_ph = tf.placeholder(tf.float32, [vocabulary_size, embedding_dimension])
seq_lens_ph = tf.placeholder(tf.int64, [batch_size, ])

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
    for epoch in range(num_epoch):
        for batch in range(num_batch_in_epoch):
            batch_data, batch_labels, batch_seq_lens = data_interface.get_batch(dataset_type='training')
            sess.run(model.optimize, feed_dict={data_ph: batch_data,
                                                labels_ph: batch_labels,
                                                seq_lens_ph: batch_seq_lens})
        error = sess.run(model.error, feed_dict={data_ph: batch_data,
                                                 labels_ph: batch_labels,
                                                 seq_lens_ph: batch_seq_lens})
        print('Test error {:6.2f}%'.format(100 * error))
