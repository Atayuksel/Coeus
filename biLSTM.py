import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import nltk
from itertools import combinations
import math
import pickle
import os
import progressbar

import biocreative as bc

# baseline or edit distance
METHOD = 'BASELINE'

# multiclass or binary
MODE = 'BINARY'

# tensorflow error loggingos.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# dataset parameters
WORD_EMBEDDING_SIZE = 300
ED_NUM_CLASS = 14

if MODE == 'BINARY':
    NUM_CLASS = 2
elif MODE == 'MULTICLASS':
    NUM_CLASS = 6

# model parameters
NUM_EPOCH = 50
BATCH_SIZE = 100
LEARNING_RATE = 0.001
NUM_HIDDEN = 128
TAG_EMBEDDING_SIZE = 20

# file directories
PRE_TRAINED_GLOVE_LOCATION = "glove.6B/glove.6B." + str(WORD_EMBEDDING_SIZE) + "d.txt"

TRA_BASE_LOCATION = "ChemProt_Corpus/chemprot_training/chemprot_training/"
TRA_ABSTRACT_LOCATION = TRA_BASE_LOCATION + "chemprot_training_abstracts.tsv"
TRA_ENTITIES_LOCATION = TRA_BASE_LOCATION + "chemprot_training_entities.tsv"
TRA_RELATION_LOCATION = TRA_BASE_LOCATION + "chemprot_training_relations.tsv"

DEV_BASE_LOCATION = "ChemProt_Corpus/chemprot_development/chemprot_development/"
DEV_ABSTRACT_LOCATION = DEV_BASE_LOCATION + "chemprot_development_abstracts.tsv"
DEV_ENTITIES_LOCATION = DEV_BASE_LOCATION + "chemprot_development_entities.tsv"
DEV_RELATION_LOCATION = DEV_BASE_LOCATION + "chemprot_development_relations.tsv"

TEST_BASE_LOCATION = "ChemProt_Corpus/chemprot_test_gs/chemprot_test_gs/"
TEST_ABSTRACT_LOCATION = TEST_BASE_LOCATION + "chemprot_test_abstracts_gs.tsv"
TEST_ENTITIES_LOCATION = TEST_BASE_LOCATION + "chemprot_test_entities_gs.tsv"
TEST_RELATION_LOCATION = TEST_BASE_LOCATION + "chemprot_test_relations_gs.tsv"

PROTEIN_NAME_LOCATION = "dataset/full_protein_names.pkl"
PROTEIN_LABEL_LOCATION = "dataset/full_protein_labels.pkl"

TENSORBOARD_OUT = "./tensorboard"


def save_to_pickle(data, file_name):
    """
    save python data to pickle file to 'relation_extraction_pickle folder
    :param data: data to be save
    :param file_name: file_name for pickle file
    :return: None
    """

    directory = "relation_extraction_pickle/" + file_name + ".pickle"
    with open(directory, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)


def load_from_pickle(file_name):
    """
    load the file from the relation_extraction_pickle folder
    :param file_name: file name to be loaded
    :return:
    data: python data to be loaded
    """

    directory = "relation_extraction_pickle/" + file_name + ".pickle"
    with open(directory, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    return data


def check_file(file_name):
    """
    check the file if it exists
    :param file_name: file to be checked
    :return: boolean if file exists
    """

    directory = "relation_extraction_pickle/" + file_name + ".pickle"
    return os.path.isfile(directory)


def create_position_embed_map(max_value):
    """
    Creates a position id mapping based max_value parameter
    :param max_value: maximum value for mapping
    :return: position_id_mapping
    """

    res_mapping = {0: 0}
    for index in range(max_value):
        current_available_id = len(res_mapping)
        res_mapping[index] = current_available_id
        current_available_id = current_available_id + 1
        res_mapping[-index] = current_available_id

    return res_mapping


def get_training_dataset(biocreative_dataset):
    """
    obtain training dataset from biocreative
    :return:
    res_dataset: list of instances(instance dict)
    res_labels: list of labels(int)
    res_word_freq: word frequency dictionary
    res_pos_mapping: pos tags to id mapping
    res_iob_mapping: iob tags to id mapping
    res_max_length: length of the maximum sentence
    """

    if check_file('training_dataset'):
        res_dataset = load_from_pickle('training_dataset')
        res_labels = load_from_pickle('training_labels')
        res_word_freq = load_from_pickle('word_freq')
        res_pos_mapping = load_from_pickle('pos_mapping')
        res_iob_mapping = load_from_pickle('iob_mapping')
        res_max_length = load_from_pickle('max_length')
    else:
        res_dataset, res_labels, res_word_freq, res_pos_mapping, res_iob_mapping, res_max_length\
            = biocreative_dataset.create_biocreative_dataset(data=0)
        save_to_pickle(res_dataset, 'training_dataset')
        save_to_pickle(res_labels, 'training_labels')
        save_to_pickle(res_word_freq, 'word_freq')
        save_to_pickle(res_pos_mapping, 'pos_mapping')
        save_to_pickle(res_iob_mapping, 'iob_mapping')
        save_to_pickle(res_max_length, 'max_length')

    res_position_id_mapping = create_position_embed_map(res_max_length)
    result = [res_dataset, res_labels, res_word_freq,
              res_pos_mapping, res_iob_mapping, res_max_length,
              res_position_id_mapping]

    return result


def get_development_dataset(biocreative_dataset):
    """
    obtain development dataset from biocreative
    :return:
    res_dataset: list of instances(instance dict)
    res_labels: list of labels(int)
    """

    if check_file('development_dataset'):
        res_dataset = load_from_pickle('development_dataset')
        res_labels = load_from_pickle('development_labels')
    else:
        res_dataset, res_labels = biocreative_dataset.create_biocreative_dataset(data=1)
        save_to_pickle(res_dataset, 'development_dataset')
        save_to_pickle(res_labels, 'development_labels')

    result = [res_dataset, res_labels]
    return result


def get_test_dataset(biocreative_dataset):
    """
    obtain test dataset from biocreative
    :return:
    res_dataset: list of instances(instance dict)
    res_labels: list of labels(int)
    """

    if check_file('test_dataset'):
        res_dataset = load_from_pickle('test_dataset')
        res_labels = load_from_pickle('test_labels')
    else:
        res_dataset, res_labels = biocreative_dataset.create_biocreative_dataset(data=2)
        save_to_pickle(res_dataset, 'test_dataset')
        save_to_pickle(res_labels, 'test_labels')

    result = [res_dataset, res_labels]
    return result


def baseline_update_embeddings(word_id_mapping, embedding_list, embedding_dim):
    """
    Update the word embedding list with respect to baseline method.
    Add zero vector to embedding list
    Add 'CHEMICAL' and 'PROTEIN' to the embedding dictionary
    :param word_id_mapping: (dictionary) mapping of word to unique id
    :param embedding_list: list of embedding numpy
    :param embedding_dim: selected embedding dimension
    :return:
    word_id_mapping: updated mapping of word to unique id
    embedding_list: updated list of embedding numpy
    """

    for i in range(2):
        word_embedding = np.zeros(embedding_dim)
        embedding_list.append(word_embedding)
        if i == 0:
            token_name = "CHEMICAL"
        else:
            token_name = "PROTEIN"
        word_id_mapping[token_name] = len(word_id_mapping)
    return word_id_mapping, embedding_list


def edit_distance_update_embeddings(word_id_mapping, embedding_list, embedding_dim):
    """
    Update the word embedding list with respect to edit distance method.
    Add zero vector to embedding list
    Add 'CHEMICAL' and token for each protein class. For example; "PROTEIN1"
    :param word_id_mapping: (dictionary) mapping of word to unique id
    :param embedding_list: list of embedding numpy
    :param embedding_dim: selected embedding dimension
    :return:
    word_id_mapping: updated mapping of word to unique id
    embedding_list: updated list of embedding numpy
    """

    word_embedding = np.zeros(embedding_dim)
    embedding_list.append(word_embedding)
    token_name = "CHEMICAL"
    word_id_mapping[token_name] = len(word_id_mapping)
    for i in range(ED_NUM_CLASS):
        word_embedding = np.zeros(embedding_dim)
        embedding_list.append(word_embedding)
        token_name = "PROTEIN" + str(i)
        word_id_mapping[token_name] = len(word_id_mapping)
    return word_id_mapping, embedding_list


def create_embedding_matrix(file_location):
    """
    Takes pre defined word embeddings file location and creates embeddings matrix
    :param file_location: GLOVE file location
    :return:
    res_embedding_list: numpy array (vocab_size x embedding_size)
    res_word_id_mapping: mapping words to unique ids
    """

    res_word_id_mapping = {}
    res_embedding_list = []

    # parse embedding file
    with open(file_location, 'r', encoding='utf-8') as embedding_file:
        line = embedding_file.readline()[:-1]
        while line:
            # parse the line
            line = line.split(' ')

            # get token
            token = line[0]

            # get embedding and convert it to numpy array
            word_embedding = line[1:]
            word_embedding = list(np.float_(word_embedding))
            word_embedding = np.asarray(word_embedding)

            # add embedding to embedding list
            res_embedding_list.append(word_embedding)

            # assign id to token
            current_id = len(res_word_id_mapping)
            res_word_id_mapping[token] = current_id

            # read new line
            line = embedding_file.readline()[:-1]

    # add embeddings based on the selected method
    embedding_dim = res_embedding_list[0].shape[0]

    if METHOD == 'BASELINE':
        res_word_id_mapping, res_embedding_list = baseline_update_embeddings(res_word_id_mapping,
                                                                             res_embedding_list,
                                                                             embedding_dim)
    elif METHOD == 'EDITDISTANCE':
        res_word_id_mapping, res_embedding_list = edit_distance_update_embeddings(res_word_id_mapping,
                                                                                  res_embedding_list,
                                                                                  embedding_dim)

    # vertical stack all embeddings and create a numpy array
    res_embedding_list = np.vstack(res_embedding_list)
    return res_embedding_list, res_word_id_mapping


def get_embeddings():
    """
    check if embeddings are also prepared.
    if they are already processed then load the file
    else prepare embeddings from scratch
    :return:
    res_embedding: numpy embedding matrix (vocab_size x embedding_size)
    res_word_idx_mapping: mapping of words to unique ids
    """

    if check_file('embedding'):
        res_embedding = load_from_pickle('embedding')
        res_word_idx_mapping = load_from_pickle('word_idx_mapping')
    else:
        res_embedding, res_word_idx_mapping = create_embedding_matrix(PRE_TRAINED_GLOVE_LOCATION)
        save_to_pickle(res_embedding, 'embedding')
        save_to_pickle(res_word_idx_mapping, 'word_idx_mapping')
    return res_embedding, res_word_idx_mapping


def create_protein_dictionary(name_list, label_list):
    """
    creates a dictionary from two files.
    each protein label is key, and protein names are list.
    :param name_list: list protein names
    :param label_list: list protein family class id
    :return: protein label dictionary
    """

    result = {}
    for i in range(len(name_list)):
        protein_name = name_list[i]
        protein_label = label_list[i]
        if protein_label not in result:
            result[protein_label] = []
        result[protein_label].append(protein_name)
    return result


def dense_ref_data(reference_data, limit):
    """
    takes reference data edit distance and concatenate smaller families under limit
    :param reference_data: reference data dictionary
    :param limit: limit for selecting available families
    :return result: updated reference data dictionary
    """

    families_over_limit = []
    out_proteins = []
    for key, value_list in reference_data.items():
        if len(value_list) > limit:
            families_over_limit.append(value_list)
        else:
            for value in value_list:
                out_proteins.append(value)

    # create protein family classes
    result = {}
    cur_key = 0
    for tmp_list in families_over_limit:
        result[cur_key] = tmp_list
        cur_key = cur_key + 1
    result[cur_key] = out_proteins

    return result


def create_sparse_vec(word_list):
    """
    creates tensorflow sparse vector for edit distance operation
    :param word_list: list of words
    :return: tensorflow sparse vector
    """

    num_words = len(word_list)
    indices = [[xi, 0, yi] for xi, x in enumerate(word_list) for yi, y in enumerate(x)]
    chars = list(''.join(word_list))
    return tf.SparseTensorValue(indices, chars, [num_words, 1, 1])


def get_batch_data(batch_size, index, data, label_data=None):
    """
    get total dataset and return batch data.
    :param batch_size: selected batch size
    :param index: current offset
    :param data: dataset input
    :param label_data: dataset labels
    :return batch_input: input of the next batch
    :return batch_label: label of the next batch
    :return index: updated offset in the dataset
    """

    # set index if index is end
    if index == len(data):
        index = 0

    batch_input = data[index:min(index+batch_size, len(data))]

    if label_data:
        batch_label = label_data[index:min(index+batch_size, len(data))]
        index = min((index+batch_size), len(data))
        return batch_input, batch_label, index
    else:
        index = min((index+batch_size), len(data))
        return batch_input, index


def replace_tokens(instance, protein_family_mapping):
    """
    get sentence and replace arguments with selected tokens
    obtain argument indexes
    :param instance: dataset instance
    :param protein_family_mapping: mapping of protein names to protein family classes
    :return tokens: tokenized list of instance
    :return argument_idx: return indexes of the arguments in tokens list
    """

    sentence = instance[1]

    arg1_text = instance[3]
    arg1_type = instance[4]

    arg2_text = instance[6]
    arg2_type = instance[7]

    replaced_words = []

    # replace arg1
    if arg1_type == 'CHEMICAL':
        sentence = sentence.replace(arg1_text, 'CHEMICAL')
        replaced_words.append('CHEMICAL')
    else:
        if METHOD == 'EDITDISTANCE':
            selected_protein_token = protein_family_mapping[arg1_text]
            sentence = sentence.replace(arg1_text, selected_protein_token)
            replaced_words.append(selected_protein_token)
        elif METHOD == 'BASELINE':
            sentence = sentence.replace(arg1_text, 'PROTEIN')
            replaced_words.append('PROTEIN')

    # replace arg2
    if arg2_type == 'CHEMICAL':
        sentence = sentence.replace(arg2_text, 'CHEMICAL')
        replaced_words.append('CHEMICAL')
    else:
        if METHOD == 'EDITDISTANCE':
            selected_protein_token = protein_family_mapping[arg2_text]
            sentence = sentence.replace(arg2_text, selected_protein_token)
            replaced_words.append(selected_protein_token)
        elif METHOD == 'BASELINE':
            sentence = sentence.replace(arg2_text, 'PROTEIN')
            replaced_words.append('PROTEIN')

    # check and fix tokens
    argument_idx = []
    tokens = nltk.word_tokenize(sentence)
    for index in range(len(tokens)):
        token = tokens[index]
        for new_word in replaced_words:
            if new_word in token:
                tokens[index] = new_word
                argument_idx.append(index)

    argument_idx = argument_idx[:2]
    # assert len(argument_idx) == 2, tokens

    if len(argument_idx) == 1:
        argument_idx.append(np.random.randint(len(tokens)))
    if len(argument_idx) == 0:
        argument_idx.append(np.random.randint(len(tokens)))
        argument_idx.append(np.random.randint(len(tokens)))

    return tokens, argument_idx


def convert_input(data, word_mapping, pos_mapping, iob_mapping, distance_mapping, max_sent_len, prot_label_dict):
    """
    get list of instances and prepare input tensor for lstm
    :param data: list of instances
    :param word_mapping: mapping of word tokens to id
    :param pos_mapping: mapping of pos tokens to id
    :param iob_mapping: mapping of iob tokens to id
    :param distance_mapping: mapping of distance to id
    :param max_sent_len: maximum sentence length
    :param prot_label_dict: protein name to label dictionary
    :return result: array of ids
    """

    result_word_idx = []
    result_pos_idx = []
    result_iob_idx = []
    result_arg1_distance_idx = []
    result_arg2_distance_idx = []
    result_sentence_length = []

    # for each instance in data
    for instance in data:
        instance_word_ids = np.zeros(max_sent_len, dtype=np.int64)
        instance_pos_tags = np.zeros(max_sent_len, dtype=np.int64)
        instance_iob_tags = np.zeros(max_sent_len, dtype=np.int64)
        instance_arg1_distance = np.zeros(max_sent_len, dtype=np.int64)
        instance_arg2_distance = np.zeros(max_sent_len, dtype=np.int64)

        # tokenize sentence, replace argument names, get indexes of arguments
        instance_tokens, arguments_indexes = replace_tokens(instance, prot_label_dict)

        pos_tags = nltk.pos_tag(instance_tokens)
        chunk_tree = nltk.ne_chunk(pos_tags)
        iob_tags = nltk.tree2conlltags(chunk_tree)

        for idx in range(len(iob_tags)):
            if idx < max_sent_len:
                tags = iob_tags[idx]
                word_tag = tags[0]
                pos_tag = tags[1]
                iob_tag = tags[2]
                distance_to_arg1 = arguments_indexes[0] - idx
                distance_to_arg2 = arguments_indexes[1] - idx

                if word_tag.lower() in word_mapping:
                    token_id = word_mapping[word_tag.lower()]
                else:
                    token_id = word_mapping['unk']

                if pos_tag in pos_mapping:
                    pos_id = pos_mapping[pos_tag]
                else:
                    pos_id = pos_mapping['unk']

                if iob_tag in iob_mapping:
                    iob_id = iob_mapping[iob_tag]
                else:
                    iob_id = iob_mapping['unk']

                distance_arg1_id = distance_mapping[distance_to_arg1]
                distance_arg2_id = distance_mapping[distance_to_arg2]

                instance_word_ids[idx] = int(token_id)
                instance_pos_tags[idx] = int(pos_id)
                instance_iob_tags[idx] = int(iob_id)
                instance_arg1_distance[idx] = int(distance_arg1_id)
                instance_arg2_distance[idx] = int(distance_arg2_id)

        result_word_idx.append(instance_word_ids)
        result_pos_idx.append(instance_pos_tags)
        result_iob_idx.append(instance_iob_tags)
        result_arg1_distance_idx.append(instance_arg1_distance)
        result_arg2_distance_idx.append(instance_arg2_distance)
        result_sentence_length.append(len(iob_tags))

    result_word_idx = np.vstack(result_word_idx)
    result_pos_idx = np.vstack(result_pos_idx)
    result_iob_idx = np.vstack(result_iob_idx)
    result_arg1_distance_idx = np.vstack(result_arg1_distance_idx)
    result_arg2_distance_idx = np.vstack(result_arg2_distance_idx)
    result_sentence_length = np.asarray(result_sentence_length)

    result_arr = [result_word_idx, result_pos_idx, result_iob_idx, result_arg1_distance_idx,
                  result_arg2_distance_idx, result_sentence_length]

    return result_arr


def convert_labels(data, size):
    """
    convert labels to one-hot representation for lstm.
    :param data: list of labels
    :param size: size of one-hot representation
    :return result: numpy array of labels one-hot representation
    """

    result = []

    if MODE == 'BINARY':
        for instance_label in data:
            one_hot_instance = np.zeros(size)
            if instance_label == 3 or instance_label == 4 or instance_label == 5 or \
                    instance_label == 6 or instance_label == 9:
                one_hot_instance[1] = 1
            else:
                one_hot_instance[0] = 1
            result.append(one_hot_instance)

    if MODE == 'MULTICLASS':
        for instance_label in data:
            one_hot_instance = np.zeros(size)
            if instance_label == 3:
                one_hot_instance[1] = 1
            elif instance_label == 4:
                one_hot_instance[2] = 1
            elif instance_label == 5:
                one_hot_instance[3] = 1
            elif instance_label == 6:
                one_hot_instance[4] = 1
            elif instance_label == 9:
                one_hot_instance[5] = 1
            else:
                one_hot_instance[0] = 1
            result.append(one_hot_instance)

    result = np.vstack(result)
    return result


def fetch_embeddings(word_ids, pos_ids, iob_ids, distance_arg1_ids, distance_arg2_ids, embeddings_variable):
    """
    fetch embeddings of the ids. create a embedding matrix
    :param word_ids: list of word ids
    :param pos_ids: list of pos tag ids
    :param iob_ids: list of iob tag ids
    :param distance_arg1_ids: list of distance embedding ids wrt arg1
    :param distance_arg2_ids: list of distance embedding ids wrt arg2
    :param embeddings_variable: tensorflow variable that stores embeddings
    :return sentence_image: concatenated matrix of embeddings
    """

    sentence_word_image = tf.nn.embedding_lookup(embeddings_variable['word'], word_ids)
    sentence_pos_image = tf.nn.embedding_lookup(embeddings_variable['pos'], pos_ids)
    sentence_iob_image = tf.nn.embedding_lookup(embeddings_variable['iob'], iob_ids)
    sentence_distance_arg1_image = tf.nn.embedding_lookup(embeddings_variable['arg1_distance'], distance_arg1_ids)
    sentence_distance_arg2_image = tf.nn.embedding_lookup(embeddings_variable['arg2_distance'], distance_arg2_ids)

    sentence_image = tf.concat(axis=2, values=[sentence_word_image, sentence_pos_image, sentence_iob_image,
                                               sentence_distance_arg1_image, sentence_distance_arg2_image])
    return sentence_image


def model(x, seq_len_list, max_seq_len):
    """
    model of bidirectional lstm with dropout wrapper
    :param x: input (batch_size x time_step x embedding_size)
    :param seq_len_list: list of sentence lengths
    :return lstm_output: output of the model (time_step x batch_size x num_hidden)
    """

    # convert input to time_step x batch_size x embedding_size
    x = tf.unstack(x, max_seq_len, 1)

    lstm_fw_cell = rnn.LSTMBlockCell(num_units=NUM_HIDDEN)
    lstm_fw_cell_dropout = rnn.DropoutWrapper(cell=lstm_fw_cell,
                                              input_keep_prob=prob_placeholder,
                                              output_keep_prob=prob_placeholder,
                                              state_keep_prob=prob_placeholder)

    lstm_bw_cell = rnn.LSTMBlockCell(num_units=NUM_HIDDEN)
    lstm_bw_cell_dropout = rnn.DropoutWrapper(cell=lstm_bw_cell,
                                              input_keep_prob=prob_placeholder,
                                              output_keep_prob=prob_placeholder,
                                              state_keep_prob=prob_placeholder)

    lstm_output, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell_dropout, lstm_bw_cell_dropout, x,
                                                     sequence_length=seq_len_list,
                                                     dtype=tf.float32)

    return lstm_output


def lstm_max_pooling(lstm_output, seq_length_list):
    """
    apply max_pooling over sequence to bi-lstm output.
    :param lstm_output: output of the lstm layer output
    :param seq_length_list: list of sentence lengths
    :return output: output after applying max pooling (batch_size x num_hidden)
    """

    batch_outputs = tf.stack(values=lstm_output, axis=1)
    result = []

    for i in range(batch_outputs.shape[0]):
        sentence_length = seq_length_list[i]
        output = batch_outputs[i]
        output = output[:sentence_length]
        output = tf.reduce_max(input_tensor=output,
                               axis=0,
                               keepdims=False)
        result.append(output)

    output = tf.stack(values=result, axis=0)
    return output


def update_confusion_matrix(confusion, predict, true):
    """
    update confusion matrix
    :param confusion: confusion matrix
    :param predict: list of predicted values
    :param true: list of true values
    :return confusion: updated confusion matrix
    """

    true_labels = np.argmax(a=true, axis=1)
    for index in range(len(predict)):
        predicted_label = predict[index]
        true_label = true_labels[index]
        confusion[predicted_label][true_label] = confusion[predicted_label][true_label] + 1

    return confusion


def print_confusion_matrix(confusion):
    """
    prints the confusion matrix
    :param confusion: confusion matrix
    :return None:
    """

    true_value_distribution = np.sum(confusion, axis=0)
    predicted_value_distribution = np.sum(confusion, axis=1)

    for i in range(NUM_CLASS):
        num_prediction = predicted_value_distribution[i]
        num_true = true_value_distribution[i]
        TP = confusion[i][i]
        FP = num_prediction - TP
        FN = num_true - TP
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f_measure = 2 * ((precision*recall)/(precision+recall))
        print("(Category {}) - precision:{}, recall:{}, f-measure:{}".format(i, precision, recall, f_measure))


def calculate_f_measure(confusion_matrix):

    result = []

    true_value_distribution = np.sum(confusion_matrix, axis=0)
    predicted_value_distribution = np.sum(confusion_matrix, axis=1)

    tp_values = []
    fp_values = []
    fn_values = []

    precision_values = []
    recall_values = []
    f_measure_values = []

    for i in range(NUM_CLASS):
        num_prediction = predicted_value_distribution[i]
        num_true = true_value_distribution[i]
        TP = confusion_matrix[i][i]
        FP = num_prediction - TP
        FN = num_true - TP
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f_measure = 2 * ((precision*recall)/(precision+recall))
        tp_values.append(TP)
        fp_values.append(FP)
        fn_values.append(FN)
        precision_values.append(precision)
        recall_values.append(recall)
        f_measure_values.append(f_measure)

    micro_precision = sum(tp_values) / (sum(tp_values) + sum(fp_values))
    macro_precision = sum(precision_values) / len(precision_values)

    micro_recall = sum(tp_values) / (sum(tp_values) + sum(fn_values))
    macro_recall = sum(recall_values) / len(recall_values)

    micro_f_measure = 2 * ((micro_precision*micro_recall)/(micro_precision+micro_recall))
    macro_f_measure = sum(f_measure_values)/len(f_measure_values)

    result = [micro_precision, micro_recall, micro_f_measure, macro_precision, macro_recall, macro_f_measure]
    return result


def run_edit_distance_graph(abstract_entities_dict, protein_family_dict):
    """
    run edit distance graph and obtain mapping of protein names to family labels
    :param abstract_entities_dict: mapping of abstracts to entities
    :param protein_family_dict: reference data for edit distance
    :return result: mapping of protein names to protein families
    """

    result = {}
    with tf.Session(graph=edit_distance_graph) as edit_sess:
        edit_distance_bar = progressbar.ProgressBar(maxval=len(abstract_entities_dict),
                                                    widgets=[progressbar.Bar('=', '[', ']'),
                                                             ' ',
                                                             progressbar.Percentage()])
        edit_distance_bar_counter = 0
        edit_distance_bar.start()

        # for each abstract
        for abstract_id, bio_entities in abstract_entities_dict.items():
            # for each entity in the abstract
            for bio_entity in bio_entities:
                if bio_entity['entityType'] != 'CHEMICAL':
                    bio_entity_name = bio_entity['name']
                    test_string = [bio_entity_name]
                    distance_values = np.zeros(ED_NUM_CLASS)  # create an array that stores distance to each family
                    # for each protein family
                    for i in range(ED_NUM_CLASS):
                        ref_strings = protein_family_dict[i]  # get proteins in ith family

                        # create vectors for tf.edit_distance func
                        test_string_sparse = create_sparse_vec(test_string * len(ref_strings))
                        ref_strings_sparse = create_sparse_vec(ref_strings)

                        feed_dict = {test_word_placeholder: test_string_sparse,
                                     ref_words_placeholder: ref_strings_sparse}

                        # distance to ith family
                        distance = edit_sess.run(min_distance, feed_dict=feed_dict)
                        # set distance array
                        distance_values[i] = distance

                    # get index of min value in distance values array
                    selected_label = np.argmin(distance_values)
                    # create a label for the protein
                    selected_label = "PROTEIN" + str(selected_label)
                    # set the result dictionary
                    result[bio_entity_name] = selected_label

            edit_distance_bar_counter = edit_distance_bar_counter + 1
            edit_distance_bar.update(edit_distance_bar_counter)

        edit_distance_bar.finish()

    return result


def run_model(session, num_step, data_input, data_output, mapping_dict, train):

    # create a confusion matrix to store evaluation for each epoch
    confusion_matrix = np.zeros(shape=(NUM_CLASS, NUM_CLASS))

    # set progress bar
    bar = progressbar.ProgressBar(maxval=num_step,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    progress_bar_counter = 0
    bar.start()

    data_offset = 0
    for step in range(num_step):

        # fetch and prepare batch data
        batch_dataset, batch_labels, data_offset = get_batch_data(BATCH_SIZE,
                                                                  data_offset,
                                                                  data_input,
                                                                  data_output)

        prepared_inputs = convert_input(data=batch_dataset,
                                        word_mapping=mapping_dict['word'],
                                        pos_mapping=mapping_dict['pos'],
                                        iob_mapping=mapping_dict['iob'],
                                        distance_mapping=mapping_dict['distance'],
                                        max_sent_len=max_time_step,
                                        prot_label_dict=entity_family_mapping)

        batch_word_dataset = prepared_inputs[0]
        batch_pos_dataset = prepared_inputs[1]
        batch_iob_dataset = prepared_inputs[2]
        batch_arg1_dist = prepared_inputs[3]
        batch_arg2_dist = prepared_inputs[4]
        batch_sentence_length = prepared_inputs[5]

        # convert labels to one-hot representation
        batch_labels = convert_labels(batch_labels, NUM_CLASS)

        if len(batch_dataset) == BATCH_SIZE:
            if train:
                _, predicted_values = session.run([train_op, label_prediction],
                                                  feed_dict={word_ids_placeholder: batch_word_dataset,
                                                             pos_ids_placeholder: batch_pos_dataset,
                                                             iob_ids_placeholder: batch_iob_dataset,
                                                             arg1_position_ids_placeholder: batch_arg1_dist,
                                                             arg2_position_ids_placeholder: batch_arg2_dist,
                                                             prob_placeholder: 0.5,
                                                             Y: batch_labels,
                                                             sentence_length_placeholder: batch_sentence_length})
            else:
                predicted_values = session.run(label_prediction,
                                               feed_dict={word_ids_placeholder: batch_word_dataset,
                                                          pos_ids_placeholder: batch_pos_dataset,
                                                          iob_ids_placeholder: batch_iob_dataset,
                                                          arg1_position_ids_placeholder: batch_arg1_dist,
                                                          arg2_position_ids_placeholder: batch_arg2_dist,
                                                          prob_placeholder: 1.0,
                                                          sentence_length_placeholder: batch_sentence_length})

            confusion_matrix = update_confusion_matrix(confusion=confusion_matrix,
                                                       true=batch_labels,
                                                       predict=predicted_values)

        progress_bar_counter = progress_bar_counter + 1
        bar.update(progress_bar_counter)

    bar.finish()

    return confusion_matrix


biocreative = bc.BioCreativeData(TRA_ABSTRACT_LOCATION, TRA_ENTITIES_LOCATION, TRA_RELATION_LOCATION,
                                 DEV_ABSTRACT_LOCATION, DEV_ENTITIES_LOCATION, DEV_RELATION_LOCATION,
                                 TEST_ABSTRACT_LOCATION, TEST_ENTITIES_LOCATION, TEST_RELATION_LOCATION)

training_data = get_training_dataset(biocreative)
training_instances = training_data[0]
training_labels = training_data[1]
training_word_freq = training_data[2]
training_max_seq_len = training_data[5]

pos_mapping = training_data[3]
iob_mapping = training_data[4]
distance_mapping = training_data[6]

development_data = get_development_dataset(biocreative)
development_instances = development_data[0]
development_labels = development_data[1]

test_data = get_test_dataset(biocreative)
test_instances = test_data[0]
test_labels = test_data[1]

embedding_matrix, word_mapping = get_embeddings()

mappings = {
    'pos': pos_mapping,
    'iob': iob_mapping,
    'distance': distance_mapping,
    'word': word_mapping
}

# Edit Distance
# the name of pre-calculated edit distance file is 'total_ed_protein_classes'
filename = 'total_protein_dataset'
if check_file(filename):
    entity_family_mapping = load_from_pickle('total_protein_dataset')
    run_edit_distance = False
    print("Edit Distance file is loaded.")
else:
    entity_family_mapping = {}
    run_edit_distance = True
    print("Edit Distance file can not be loaded.")

if run_edit_distance:
    total_entity_dict = dict(biocreative.training_entities)
    total_entity_dict.update(biocreative.development_entities)
    total_entity_dict.update(biocreative.test_entities)

    with open(PROTEIN_NAME_LOCATION, 'rb') as file:
        reference_protein_names = pickle.load(file)

    with open(PROTEIN_LABEL_LOCATION, 'rb') as file:
        reference_protein_labels = pickle.load(file)

    # pre-process protein dataset for families larger than 500
    proteins_dict = create_protein_dictionary(reference_protein_names,
                                              reference_protein_labels)
    proteins_dict = dense_ref_data(proteins_dict, 500)

    # creates edit distance graph
    edit_distance_graph = tf.Graph()
    with edit_distance_graph.as_default() as g:
        ref_words_placeholder = tf.sparse_placeholder(dtype=tf.string)
        test_word_placeholder = tf.sparse_placeholder(dtype=tf.string)

        edit_distances = tf.edit_distance(test_word_placeholder, ref_words_placeholder, normalize=False)
        min_distance = tf.reduce_min(edit_distances)
    tf.reset_default_graph()

    entity_family_mapping = run_edit_distance_graph(total_entity_dict, proteins_dict)
    save_to_pickle(entity_family_mapping, 'total_ed_protein_classes')

# LSTM
training_dataset_size = len(training_instances)
development_dataset_size = len(development_instances)
test_dataset_size = len(test_instances)

training_num_step = math.ceil(training_dataset_size/BATCH_SIZE)
development_num_step = math.ceil(development_dataset_size/BATCH_SIZE)
test_num_step = math.ceil(test_dataset_size/BATCH_SIZE)

assert WORD_EMBEDDING_SIZE == embedding_matrix.shape[1]
vocab_size = embedding_matrix.shape[0]
word_embedding_size = embedding_matrix.shape[1]

tag_embedding_size = TAG_EMBEDDING_SIZE
pos_tag_size = len(pos_mapping)
iob_tag_size = len(iob_mapping)
position_tag_size = len(distance_mapping)

max_time_step = training_max_seq_len
batch_offset = 0

# LSTM Model
# input placeholders
word_ids_placeholder = tf.placeholder("int64", [None, max_time_step])
pos_ids_placeholder = tf.placeholder("int64", [None, max_time_step])
iob_ids_placeholder = tf.placeholder("int64", [None, max_time_step])
arg1_position_ids_placeholder = tf.placeholder("int64", [None, max_time_step])
arg2_position_ids_placeholder = tf.placeholder("int64", [None, max_time_step])
sentence_length_placeholder = tf.placeholder("int64", [BATCH_SIZE, ])
Y = tf.placeholder("float", [None, NUM_CLASS])

# dropout keep probability
prob_placeholder = tf.placeholder_with_default(1.0, shape=())

# initializer placeholder for embedding variable
word_embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, word_embedding_size])

# lstm variables
weights = {
    'out': tf.Variable(tf.random_normal([2*NUM_HIDDEN, NUM_CLASS]))
}

biases = {
    'out': tf.Variable(tf.random_normal([NUM_CLASS]))
}

embeddings = {
    'pos': tf.Variable(tf.random_uniform(shape=[pos_tag_size, tag_embedding_size],
                                         maxval=1.0),
                       trainable=True,
                       name="pos_embedding_variable",
                       dtype=tf.float32),
    'iob': tf.Variable(tf.random_uniform(shape=[iob_tag_size, tag_embedding_size],
                                         minval=0.0,
                                         maxval=1.0),
                       trainable=True,
                       name="iob_embedding_variable",
                       dtype=tf.float32),
    'arg1_distance': tf.Variable(tf.random_uniform(shape=[iob_tag_size, tag_embedding_size],
                                                   minval=0.0,
                                                   maxval=1.0),
                                 trainable=True,
                                 name="iob_embedding_variable",
                                 dtype=tf.float32),
    'arg2_distance': tf.Variable(tf.random_uniform(shape=[position_tag_size, tag_embedding_size],
                                                   minval=0.0,
                                                   maxval=1.0),
                                 trainable=True,
                                 name="arg2_position_embedding_variable",
                                 dtype=tf.float32),
    'word': tf.Variable(tf.constant(0.0, shape=[vocab_size, word_embedding_size]),
                        trainable=True,
                        name="word_embedding_variable",
                        dtype=tf.float32)
}

# weights for weighted cross entropy
if MODE == 'MULTICLASS':
    class_weights = tf.constant([[1.09, 61.2, 20.86, 271.9, 200.1, 64.7]])
elif MODE == 'BINARY':
    class_weights = tf.constant([[1.54, 3]])

# fetch the embeddings of ids
batch_sentence_image = fetch_embeddings(word_ids_placeholder, pos_ids_placeholder, iob_ids_placeholder,
                                        arg1_position_ids_placeholder, arg2_position_ids_placeholder, embeddings)

# run through the model
outputs = model(batch_sentence_image, sentence_length_placeholder, max_time_step)

# apply max pooling to lstm output
max_pool_output = lstm_max_pooling(lstm_output=outputs, seq_length_list=sentence_length_placeholder)

# run through fully connected layer
logits = tf.matmul(max_pool_output, weights['out']) + biases['out']

# calculating loss
weights = tf.reduce_sum(class_weights * Y, axis=1)
unweighted_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
weighted_loss = unweighted_loss_op * weights
loss = tf.reduce_mean(weighted_loss)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,
                                   beta1=0.9,
                                   beta2=0.999)
train_op = optimizer.minimize(loss)

# predictions
prediction = tf.nn.softmax(logits)
label_prediction = tf.argmax(prediction, axis=1)

# initialization operations
init = tf.global_variables_initializer()
word_embedding_init = embeddings['word'].assign(word_embedding_placeholder)

# tensorboard writer
summary_writer = tf.summary.FileWriter(TENSORBOARD_OUT)

with tf.Session() as sess:

    sess.run(init)
    emb = sess.run(word_embedding_init, feed_dict={word_embedding_placeholder: embedding_matrix})

    saver = tf.train.Saver([embeddings['word']])

    for epoch in range(NUM_EPOCH):

        summary = tf.Summary()

        epoch_confusion_matrix = run_model(session=sess,
                                           num_step=training_num_step,
                                           data_input=training_instances,
                                           data_output=training_labels,
                                           mapping_dict=mappings,
                                           train=True)
        print_confusion_matrix(epoch_confusion_matrix)
        training_evaluation = calculate_f_measure(epoch_confusion_matrix)
        training_f_measure = training_evaluation[2]
        summary.value.add(tag='training_f_measure', simple_value=training_f_measure)

        epoch_confusion_matrix = run_model(session=sess,
                                           num_step=development_num_step,
                                           data_input=development_instances,
                                           data_output=development_labels,
                                           mapping_dict=mappings,
                                           train=False)
        development_evaluation = calculate_f_measure(epoch_confusion_matrix)
        development_f_measure = development_evaluation[2]
        summary.value.add(tag='development_f_measure', simple_value=development_f_measure)

        summary_writer.add_summary(summary, epoch)

    saver.save(sess, os.path.join(TENSORBOARD_OUT, 'images.ckpt'))
    summary_writer.flush()

    # run test dataset on trained bi-lstm
    test_confusion_matrix = run_model(session=sess,
                                      num_step=test_num_step,
                                      data_input=test_instances,
                                      data_output=test_labels,
                                      mapping_dict=mappings,
                                      train=True)
    print_confusion_matrix(test_confusion_matrix)

    # saving embedding tensor for projection
    word_embedding_matrix = sess.run(embeddings['word'])
    inv_map = {v: k for k, v in mappings['word'].items()}
    embedding_labels = []
    for i in range(len(inv_map)):
        label = inv_map[i]
        embedding_labels.append(label)
    np.savetxt("embeddings.tsv", word_embedding_matrix, delimiter="\t")
    with open('embedding_labels.tsv', 'w') as f:
        for label in embedding_labels:
            f.write("%s\n" % label)
