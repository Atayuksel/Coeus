import tensorflow as tf
import numpy as np
import coeus

TRAINING_PROTEINFILE_LOCATION = "dataset/train_data.pkl"
TRAINING_LABEL_LOCATION = "dataset/train_label.pkl"
TEST_PROTEINFILE_LOCATION = "dataset/test_data.pkl"
TEST_LABEL_LOCATION = "dataset/test_label.pkl"

dataset = coeus.WordDataset(TRAINING_PROTEINFILE_LOCATION,
                            TEST_PROTEINFILE_LOCATION,
                            TRAINING_LABEL_LOCATION,
                            TEST_LABEL_LOCATION)

SELECTION = "min"
NUM_CLASS = dataset.getclasslength()

def create_sparse_vec(word_list):
    num_words = len(word_list)
    indices = [[xi, 0, yi] for xi, x in enumerate(word_list) for yi, y in enumerate(x)]
    chars = list(''.join(word_list))
    return tf.SparseTensorValue(indices, chars, [num_words, 1, 1])

ref_words_placeholder = tf.sparse_placeholder(dtype=tf.string)
test_word_placeholder = tf.sparse_placeholder(dtype=tf.string)

edit_distances = tf.edit_distance(test_word_placeholder, ref_words_placeholder, normalize=False)
min_distance = tf.reduce_min(edit_distances)

test_words = dataset.get_test_words()
test_labels = dataset.get_test_labels()

predicted_values = np.zeros(len(test_words))

with tf.Session() as sess:
    for test_id in range(len(test_words)):
        test_term = test_words[test_id]
        test_strings = [test_term]
        distance_values = np.zeros(NUM_CLASS)
        for i in range(NUM_CLASS):
            ref_strings = dataset.get_class_words(i)

            test_string_sparse = create_sparse_vec(test_strings * len(ref_strings))
            ref_strings_sparse = create_sparse_vec(ref_strings)

            feed_dict = {test_word_placeholder: test_string_sparse,
                         ref_words_placeholder: ref_strings_sparse}

            result = sess.run(min_distance, feed_dict=feed_dict)
            distance_values[i] = result
        selected_label = np.argmin(distance_values)
        predicted_values[test_id] = selected_label