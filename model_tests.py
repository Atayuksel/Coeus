import tensorflow as tf
import numpy as np
from tensorflow import keras
import bilstm_model


def convert_one_hot(data):
    max_value = max(data)
    result_data = []
    for label in data:
        np_label = np.zeros(max_value+1)
        np_label[label] = 1
        result_data.append(np_label)
    return result_data

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
word_index = imdb.get_word_index()
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

label = convert_one_hot(train_labels)
vocabulary_size = len(word_index)
embedding_size = 10
batch_size = 20

pre_embedding = np.random.rand(vocabulary_size, embedding_size)

data = tf.placeholder(tf.int64, [batch_size, 250])
label = tf.placeholder(tf.float32, [batch_size, 2])
embedding = tf.placeholder(tf.float32, [vocabulary_size, embedding_size])
sequence_lengths = tf.placeholder(tf.int64, [batch_size, ])
class_weights = tf.constant([[1., 1.]])
num_hidden = 2
learning_rate = 0.01

model = bilstm_model.BiLSTMModel(data=data,
                                 target=label,
                                 seq_lens=sequence_lengths,
                                 class_weights=class_weights,
                                 num_hidden=num_hidden,
                                 learning_rate=learning_rate,
                                 embedding_size=embedding_size,
                                 vocab_size=vocabulary_size)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
word_embedding_init = model.embedding_v.assign(embedding)
sess.run(word_embedding_init, feed_dict={embedding: pre_embedding})




print('alksjdasjd')
