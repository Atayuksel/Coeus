import dataset
import nltk
import numpy as np
import pickle
from contextlib import redirect_stdout
import os
import sys
import configparser
import progressbar


class DataInterface(object):
    def __init__(self):

        # read parameters from configuration file config.init
        # create config file
        config = configparser.ConfigParser()
        config.read('config.ini')

        # read interface parameters
        section_name = 'INTERFACE'
        self.batch_size = int(config[section_name]['batch_size'])
        self.text_selection = config[section_name]['text_selection']
        self.relation_type = config[section_name]['relation_type']
        self.dataset_source = config[section_name]['dataset_source']
        self.root_directory = config[section_name]['root_directory']
        self.word_tokenizer = config[section_name]['word_tokenizer']
        self.word_embedding_dir = config[section_name]['word_embedding_dir']

        # flags for reading dataset
        self.read_training_set = True
        self.read_development_set = True
        self.read_test_set = True

        # read embedding parameters
        # word embedding
        section_name = 'EMBEDDINGS'

        # position embedding
        self.position_embedding_flag = config[section_name]['position_embedding_flag']
        self.position_embedding_dir = config[section_name]['position_embedding_dir']
        self.position_embedding_flag = True if self.position_embedding_flag == 'true' else False
        self.position_embedding_size = 0
        if self.position_embedding_flag:
            self.position_embedding_size = int(config[section_name]['position_embedding_size'])

        # pos tag embedding
        self.pos_tag_embedding_flag = config[section_name]['pos_tag_embedding_flag']
        self.pos_tag_embedding_dir = config[section_name]['pos_tag_embedding_dir']
        self.pos_tag_embedding_flag = True if self.pos_tag_embedding_flag == 'true' else False
        self.pos_tag_embedding_size = 0
        if self.pos_tag_embedding_flag:
            self.pos_tag_embedding_size = int(config[section_name]['pos_tag_embedding_size'])

        # iob tag embedding
        self.iob_tag_embedding_flag = config[section_name]['iob_tag_embedding_flag']
        self.iob_tag_embedding_dir = config[section_name]['iob_tag_embedding_dir']
        self.iob_tag_embedding_flag = True if self.iob_tag_embedding_flag == 'true' else False
        self.iob_tag_embedding_size = 0
        if self.iob_tag_embedding_flag:
            self.iob_tag_embedding_size = int(config[section_name]['iob_tag_embedding_size'])

        # prepared dataset
        self.dataset = {'training': {'data': [], 'pos_tags': [], 'iob_tags': [], 'labels': [], 'entities': [],
                                     'abstract_ids': [], 'entity_ids': [], 'seq_lens': [], 'false_positive': [],
                                     'false_negative': [], 'argument_locations': [], 'max_seq_len': 0, 'batch_idx': 0},
                        'development': {'data': [], 'pos_tags': [], 'iob_tags': [], 'labels': [], 'entities': [],
                                        'abstract_ids': [], 'entity_ids': [], 'seq_lens': [], 'false_positive': [],
                                        'false_negative': [], 'argument_locations': [], 'max_seq_len': 0, 'batch_idx': 0},
                        'test': {'data': [], 'pos_tags': [], 'iob_tags': [], 'labels': [], 'entities': [],
                                 'abstract_ids': [], 'entity_ids': [], 'seq_lens': [], 'false_positive': [],
                                 'false_negative': [], 'argument_locations': [], 'max_seq_len': 0, 'batch_idx': 0}}

        self.raw_dataset = []  # raw relation dataset

        # load raw dataset
        if self.dataset_source == 'ready':
            # check prepared dataset files.
            pre_training_file_dir = self.root_directory + '/cv_training_data.pkl'
            pre_dev_file_dir = self.root_directory + '/cv_development_data.pkl'
            pre_test_file_dir = self.root_directory + '/cv_test_data.pkl'

            pre_training_file_check = os.path.exists(pre_training_file_dir)
            pre_dev_file_check = os.path.exists(pre_dev_file_dir)
            pre_test_file_check = os.path.exists(pre_test_file_dir)

            # set reading flags for dataset type
            if not pre_training_file_check:
                self.read_training_set = False
            if not pre_dev_file_check:
                self.read_development_set = False
            if not pre_test_file_check:
                self.read_test_set = False

            if not pre_training_file_check and not pre_dev_file_check and not pre_test_file_check:
                print("Error: no prepared data found in root directory: {}".format(self.root_directory))
                sys.exit()

            # load prepared dataset
            if self.read_training_set:
                pre_training_data = pickle.load(open(pre_training_file_dir, "rb"))
            if self.read_development_set:
                pre_dev_data = pickle.load(open(pre_dev_file_dir, "rb"))
            if self.read_test_set:
                pre_test_data = pickle.load(open(pre_test_file_dir, "rb"))

            # modify pre-trained dataset for data interface.
            if self.read_training_set:
                train_instance, train_label = zip(*pre_training_data)
            if self.read_development_set:
                dev_instance, dev_label = zip(*pre_dev_data)
            if self.read_test_set:
                test_instance, test_label = zip(*pre_test_data)

            # create dataset
            self.raw_dataset = []
            if self.read_training_set:
                self.raw_dataset.append(train_instance)
                self.raw_dataset.append(train_label)
            if self.read_development_set:
                self.raw_dataset.append(dev_instance)
                self.raw_dataset.append(dev_label)
            if self.read_test_set:
                self.raw_dataset.append(test_instance)
                self.raw_dataset.append(test_label)

            print('Prepared dataset is loaded successfully.')

        elif self.dataset_source == 'biocreative':

            biocreative_dataset = dataset.BioCreative(root_directory=self.root_directory,
                                                      tokenizer=self.word_tokenizer,
                                                      relation_type=self.relation_type)
            self.raw_dataset = biocreative_dataset.get_dataset()
            del biocreative_dataset

            print('Biocreative dataset is loaded successfully.')

        raw_dataset_index = 0
        if self.read_training_set:
            training_candidate_relations = self.create_candidate_relation(self.raw_dataset[raw_dataset_index],
                                                                          'training')
            raw_dataset_index = raw_dataset_index + 1
            training_labels = self.raw_dataset[raw_dataset_index]
            raw_dataset_index = raw_dataset_index + 1
        if self.read_development_set:
            development_candidate_relations = self.create_candidate_relation(self.raw_dataset[raw_dataset_index],
                                                                             'development')
            raw_dataset_index = raw_dataset_index + 1
            development_labels = self.raw_dataset[raw_dataset_index]
            raw_dataset_index = raw_dataset_index + 1
        if self.read_test_set:
            test_candidate_relations = self.create_candidate_relation(self.raw_dataset[raw_dataset_index], 'test')
            raw_dataset_index = raw_dataset_index + 1
            test_labels = self.raw_dataset[raw_dataset_index]
            raw_dataset_index = raw_dataset_index + 1

        # create word embeddings
        print("Start to fetch word embeddings from the file {}".format(self.root_directory + '/' +
                                                                       self.word_embedding_dir))
        word_embedding_data = self.create_word_embeddings()
        self.word_embedding_map = word_embedding_data[0]
        self.word_embedding_matrix = word_embedding_data[1]
        self.word_embedding_dimension = word_embedding_data[2]

        # create position embeddings
        if self.position_embedding_flag:
            if self.position_embedding_dir == '':
                print('Position Embedding is selected with size {}'.format(self.position_embedding_size))
                self.position_embedding_map = self.create_position_embeddings(training_candidate_relations)
                mu, sigma = 0, 0.1
                self.position_embedding_matrix = np.random.normal(mu, sigma, [len(self.position_embedding_map),
                                                                              self.position_embedding_size])
            # TODO: else case, implement import position embeddings from file

        # create pos tag embeddings
        if self.pos_tag_embedding_flag:
            if self.pos_tag_embedding_dir == '':
                print('POS Tag is selected with size {}'.format(self.pos_tag_embedding_size))
                self.pos_tag_embedding_map = self.create_pos_tag_embeddings(training_candidate_relations)
                mu, sigma = 0, 0.1
                self.pos_tag_embedding_matrix = np.random.normal(mu, sigma, [len(self.pos_tag_embedding_map),
                                                                             self.pos_tag_embedding_size])
            # TODO: else case, implement import pos tag embeddings from file.

        # create iob tag embeddings
        if self.iob_tag_embedding_flag:
            if self.iob_tag_embedding_dir == '':
                print('IOB Tag is selected with size {}'.format(self.iob_tag_embedding_size))
                self.iob_tag_embedding_map = self.create_iob_tag_embeddings(training_candidate_relations)
                mu, sigma = 0, 0.1
                self.iob_tag_embedding_matrix = np.random.normal(mu, sigma, [len(self.iob_tag_embedding_map),
                                                                             self.iob_tag_embedding_size])
            # TODO: else case, implement import iob tag embeddings from file.

        if self.read_training_set:
            self.create_data_dictionary(training_candidate_relations, training_labels, self.dataset['training'])
        if self.read_development_set:
            self.create_data_dictionary(development_candidate_relations, development_labels,
                                        self.dataset['development'])
        if self.read_test_set:
            self.create_data_dictionary(test_candidate_relations, test_labels, self.dataset['test'])

    @staticmethod
    def create_position_embeddings(candidate_relations):
        position_embedding_map = {0: 0, 'pad': 1}
        max_length = 0
        for i in range(len(candidate_relations)):
            candidate_relation = candidate_relations[i]
            candidate_relation_tokens = candidate_relation[1]
            if len(candidate_relation_tokens) > max_length:
                max_length = len(candidate_relation_tokens)

        for i in range(max_length):
            position_embedding_map[i] = len(position_embedding_map)
            position_embedding_map[-i] = len(position_embedding_map)

        return position_embedding_map

    @staticmethod
    def create_pos_tag_embeddings(candidate_relations):
        pos_tag_embedding_map = {'pad': 0, 'unk': 1}
        for candidate_relation in candidate_relations:
            candidate_relation_tokens = candidate_relation[1]
            candidate_relation_pos_tags = [x[1] for x in candidate_relation_tokens]
            for pos_tag in candidate_relation_pos_tags:
                if pos_tag not in pos_tag_embedding_map:
                    pos_tag_embedding_map[pos_tag] = len(pos_tag_embedding_map)
        return pos_tag_embedding_map

    @staticmethod
    def create_iob_tag_embeddings(candidate_relations):
        iob_tag_embedding_map = {'pad': 0, 'unk': 1}
        for candidate_relation in candidate_relations:
            candidate_relation_tokens = candidate_relation[1]
            candidate_relation_iob_tags = [x[2] for x in candidate_relation_tokens]
            for iob_tag in candidate_relation_iob_tags:
                if iob_tag not in iob_tag_embedding_map:
                    iob_tag_embedding_map[iob_tag] = len(iob_tag_embedding_map)
        return iob_tag_embedding_map

    def create_word_embeddings(self):
        # read word embeddings from file
        word_embedding_directory = self.root_directory + '/' + self.word_embedding_dir
        word_embedding_file = open(word_embedding_directory, 'r', encoding='utf-8')
        lines = word_embedding_file.readlines()
        word_embedding_file.close()

        word_embedding_map = {'unk': 0, 'pad': 1}
        vocabulary_size = len(lines)
        word_embedding_dimension = len(lines[0][:-1].split(' ')) - 1
        word_embedding_matrix = np.zeros(((vocabulary_size + 2), word_embedding_dimension))
        word_embedding_matrix[0:1, :] = np.random.rand(1, word_embedding_dimension)

        progress_bar = progressbar.ProgressBar(maxval=len(lines),
                                               widgets=["Reading word embedding: ",
                                                        progressbar.Bar('=', '[', ']'),
                                                        ' ',
                                                        progressbar.Percentage()])
        progress_bar_counter = 0
        progress_bar.start()

        for idx in range(len(lines)):
            line = lines[idx][:-1].split(' ')
            token = line[0]

            # get embedding and convert it to numpy array
            word_embedding = line[1:]
            word_embedding = list(np.float_(word_embedding))
            word_embedding = np.asarray(word_embedding)

            # add embedding to embeddings
            word_embedding_matrix[idx+2, :] = word_embedding

            # assign id to token
            current_id = len(word_embedding_map)
            word_embedding_map[token] = current_id

            progress_bar_counter = progress_bar_counter + 1
            progress_bar.update(progress_bar_counter)

        progress_bar.finish()

        return word_embedding_map, word_embedding_matrix, word_embedding_dimension

    def create_candidate_relation(self, raw_dataset, dataset_type):
        candidate_relation_list = []

        progress_bar = progressbar.ProgressBar(maxval=len(raw_dataset),
                                               widgets=["Preparing candidate relations ({}): ".format(dataset_type),
                                                        progressbar.Bar('=', '[', ']'),
                                                        ' ',
                                                        progressbar.Percentage()])
        progress_bar_counter = 0
        progress_bar.start()

        for i in range(len(raw_dataset)):
            instance = raw_dataset[i]

            instance_txt = instance[1]
            arg1_id = instance[2]
            arg1_txt = instance[3]
            arg1_type = instance[4]
            arg1_start_idx = int(instance[5])
            arg2_id = instance[6]
            arg2_txt = instance[7]
            arg2_type = instance[8]
            arg2_start_idx = int(instance[9])

            # fetch text between entities
            if arg1_start_idx < arg2_start_idx:
                trim_start_idx = arg1_start_idx
                trim_end_idx = arg2_start_idx + len(arg2_txt)
                candidate_relation = instance_txt[trim_start_idx:trim_end_idx]
                first_entity_id = arg1_id
            else:
                trim_start_idx = arg2_start_idx
                trim_end_idx = arg1_start_idx + len(arg1_txt)
                candidate_relation = instance_txt[trim_start_idx:trim_end_idx]
                first_entity_id = arg2_id

            # create candidate relation with respect to text selection parameter
            result = instance
            if self.text_selection == 'part':
                # tokenize candidate relation
                candidate_relation_tokens = nltk.pos_tag(nltk.word_tokenize(candidate_relation))
                grammar = "NP: {<DT>?<JJ>*<NN>}"
                cp = nltk.RegexpParser(grammar)
                with redirect_stdout(open(os.devnull, "w")):
                    candidate_relation_tokens = cp.parse(candidate_relation_tokens)
                candidate_relation_tokens = nltk.tree2conlltags(candidate_relation_tokens)

                # add tokenized candidate relations to result instance
                result[1] = candidate_relation_tokens

            elif self.text_selection == 'e_part':
                # obtain candidate relation - extended part between entities
                sentence_begin = instance_txt[0:trim_start_idx]
                sentence_end = instance_txt[trim_start_idx:]
                sentence_begin_tokens = nltk.word_tokenize(sentence_begin)
                sentence_end_tokens = nltk.word_tokenize(sentence_end)

                if len(sentence_begin_tokens) > 1:
                    candidate_relation = sentence_begin_tokens[-2] + " " + sentence_begin_tokens[-1] \
                                         + " " + candidate_relation
                elif len(sentence_begin) == 1:
                    candidate_relation = sentence_begin_tokens[-1] + " " + candidate_relation

                if len(sentence_end_tokens) > 1:
                    candidate_relation = candidate_relation + " " + sentence_end_tokens[0] + " " \
                                         + sentence_end_tokens[1]
                elif len(sentence_end_tokens) == 1:
                    candidate_relation = candidate_relation + " " + sentence_end_tokens[0]

                # tokenize candidate relation
                candidate_relation_tokens = nltk.pos_tag(nltk.word_tokenize(candidate_relation))
                grammar = "NP: {<DT>?<JJ>*<NN>}"
                cp = nltk.RegexpParser(grammar)
                with redirect_stdout(open(os.devnull, "w")):
                    candidate_relation_tokens = cp.parse(candidate_relation_tokens)
                candidate_relation_tokens = nltk.tree2conlltags(candidate_relation_tokens)

                # add tokenized candidate relations to result instance
                result[1] = candidate_relation_tokens

            elif self.text_selection == 'full':
                # tokenize candidate relation - full sentence
                instance_txt_tokens = nltk.pos_tag(nltk.word_tokenize(instance_txt))
                grammar = "NP: {<DT>?<JJ>*<NN>}"
                cp = nltk.RegexpParser(grammar)
                with redirect_stdout(open(os.devnull, "w")):
                    instance_txt_tokens = cp.parse(instance_txt_tokens)
                instance_txt_tokens = nltk.tree2conlltags(instance_txt_tokens)
                # add tokenized candidate relation to result instance
                result[1] = instance_txt_tokens

            # modify result instance
            # replace arguments if they are misplaced
            if first_entity_id == arg2_id:
                tmp = result[2:6]
                result[2:6] = result[6:]
                result[6:] = tmp

            # obtain chemical and protein locations
            # TODO: location assignment for full and e-part text selection.
            if first_entity_id == arg1_id:
                if arg1_type == 'CHEMICAL':
                    chemical_location = 0
                    protein_location = len(candidate_relation_tokens) - 1
                else:
                    protein_location = 0
                    chemical_location = len(candidate_relation_tokens) - 1
            else:
                if arg2_type == 'CHEMICAL':
                    chemical_location = 0
                    protein_location = len(candidate_relation_tokens) - 1
                else:
                    protein_location = 0
                    chemical_location = len(candidate_relation_tokens) - 1

            # append chemical and protein location to result instance
            result.append(chemical_location)
            result.append(protein_location)

            # append result to candidate relation list
            candidate_relation_list.append(result)

            progress_bar_counter = progress_bar_counter + 1
            progress_bar.update(progress_bar_counter)

        progress_bar.finish()
        # return candidate relations list
        return candidate_relation_list

    @staticmethod
    def create_data_dictionary(candidate_relations, candidate_relation_labels, data_dictionary):
        for i in range(len(candidate_relations)):
            candidate_relation = candidate_relations[i]
            candidate_relation_label = candidate_relation_labels[i]

            cr_abstract_id = candidate_relation[0]

            cr_tokens = candidate_relation[1]
            cr_tokens_txt = [x[0] for x in cr_tokens]
            cr_tokens_pos_tags = [x[1] for x in cr_tokens]
            cr_tokens_iob_tags = [x[2] for x in cr_tokens]

            cr_arg1_id = candidate_relation[2]
            cr_arg1_txt = candidate_relation[3]
            cr_arg1_type = candidate_relation[4]
            cr_arg1_start_idx = candidate_relation[5]
            cr_arg2_id = candidate_relation[6]
            cr_arg2_txt = candidate_relation[7]
            cr_arg2_type = candidate_relation[8]
            cr_arg2_start_idx = candidate_relation[9]
            chemical_location = candidate_relation[10]
            protein_location = candidate_relation[11]

            if len(cr_tokens) != 0:

                # assign maximum sequence length
                if data_dictionary['max_seq_len'] < len(cr_tokens):
                    data_dictionary['max_seq_len'] = len(cr_tokens)

                data_dictionary['abstract_ids'].append(cr_abstract_id)
                data_dictionary['data'].append(cr_tokens_txt)
                data_dictionary['pos_tags'].append(cr_tokens_pos_tags)
                data_dictionary['iob_tags'].append(cr_tokens_iob_tags)
                data_dictionary['entities'].append((cr_arg1_txt, cr_arg2_txt))
                data_dictionary['entity_ids'].append((cr_arg1_id, cr_arg2_id))
                data_dictionary['labels'].append(candidate_relation_label)
                data_dictionary['seq_lens'].append(len(cr_tokens))
                data_dictionary['argument_locations'].append((protein_location, chemical_location))

    def get_batch(self, dataset_type):

        # fetch candidate relation dataset
        candidate_relation_dataset = self.dataset[dataset_type]

        # calculate start and end index of batch
        if candidate_relation_dataset['batch_idx'] == len(candidate_relation_dataset['data']):
            candidate_relation_dataset['batch_idx'] = 0
        batch_start_idx = candidate_relation_dataset['batch_idx']
        batch_end_idx = min([batch_start_idx + self.batch_size, len(candidate_relation_dataset['data'])])

        # get pad ids for embeddings
        word_embedding_pad_id = self.word_embedding_map['pad']
        if self.position_embedding_flag:
            position_embedding_pad_id = self.position_embedding_map['pad']

        if self.pos_tag_embedding_flag:
            pos_tag_embedding_pad_id = self.pos_tag_embedding_map['pad']

        if self.iob_tag_embedding_flag:
            iob_tag_embedding_pad_id = self.iob_tag_embedding_map['pad']

        # prepare matrices for batch data: word, position, pos tag and iob tag embeddings.
        batch_word_embedding_ids = np.full((self.batch_size, candidate_relation_dataset['max_seq_len']),
                                           word_embedding_pad_id)
        if self.position_embedding_flag:
            batch_distance_protein = np.full((self.batch_size, candidate_relation_dataset['max_seq_len']),
                                             position_embedding_pad_id)
            batch_distance_chemical = np.full((self.batch_size, candidate_relation_dataset['max_seq_len']),
                                              position_embedding_pad_id)
        if self.pos_tag_embedding_flag:
            batch_pos_ids = np.full((self.batch_size, candidate_relation_dataset['max_seq_len']), pos_tag_embedding_pad_id)

        if self.iob_tag_embedding_flag:
            batch_iob_ids = np.full((self.batch_size, candidate_relation_dataset['max_seq_len']), iob_tag_embedding_pad_id)

        # for each sentence from batch_start_idx to batch_end_idx
        for batch_idx in range(batch_start_idx, batch_end_idx):
            cr_tokens_txt = candidate_relation_dataset['data'][batch_idx]

            if self.pos_tag_embedding_flag:
                cr_tokens_pos = candidate_relation_dataset['pos_tags'][batch_idx]

            if self.iob_tag_embedding_flag:
                cr_tokens_iob = candidate_relation_dataset['iob_tags'][batch_idx]

            if self.position_embedding_flag:
                cr_protein_location = candidate_relation_dataset['argument_locations'][batch_idx][0]
                cr_chemical_location = candidate_relation_dataset['argument_locations'][batch_idx][1]

            # traverse sentence for each token
            for i in range(len(cr_tokens_txt)):
                token_txt = cr_tokens_txt[i]

                if self.pos_tag_embedding_flag:\
                    pos_tag = cr_tokens_pos[i]

                if self.iob_tag_embedding_flag:
                    iob_tag = cr_tokens_iob[i]

                if self.position_embedding_flag:
                    distance_to_protein = i - cr_protein_location  # token distance to protein
                    distance_to_chemical = i - cr_chemical_location  # token distance to chemical

                # obtain token txt id
                if token_txt in self.word_embedding_map:
                    token_txt_id = self.word_embedding_map[token_txt]
                else:
                    token_txt_id = self.word_embedding_map['unk']

                # obtain pos id
                if self.pos_tag_embedding_flag:
                    if pos_tag in self.pos_tag_embedding_map:
                        pos_tag_id = self.pos_tag_embedding_map[pos_tag]
                    else:
                        pos_tag_id = self.pos_tag_embedding_map['unk']

                # obtain iob tag id
                if self.iob_tag_embedding_flag:
                    if iob_tag in self.iob_tag_embedding_map:
                        iob_tag_id = self.iob_tag_embedding_map[iob_tag]
                    else:
                        iob_tag_id = self.iob_tag_embedding_map['unk']

                # adjust distance to protein in training max length
                if self.position_embedding_flag:
                    if distance_to_protein >= self.dataset['training']['max_seq_len']:
                        distance_to_protein = self.dataset['training']['max_seq_len'] - 1
                    elif distance_to_protein <= -self.dataset['training']['max_seq_len']:
                        distance_to_protein = -self.dataset['training']['max_seq_len'] + 1

                    # adjust distance to chemical in training max length
                    if distance_to_chemical >= self.dataset['training']['max_seq_len']:
                        distance_to_chemical = self.dataset['training']['max_seq_len'] - 1
                    elif distance_to_chemical <= -self.dataset['training']['max_seq_len']:
                        distance_to_chemical = -self.dataset['training']['max_seq_len'] + 1

                    # get distance to protein id mapping
                    distance_to_protein_id = self.position_embedding_map[distance_to_protein]

                    # get distance to chemical id mapping
                    distance_to_chemical_id = self.position_embedding_map[distance_to_chemical]

                # fill token information in batch matrix
                batch_word_embedding_ids[(batch_idx % self.batch_size), i] = token_txt_id
                if self.pos_tag_embedding_flag:
                    batch_pos_ids[(batch_idx % self.batch_size), i] = pos_tag_id

                if self.iob_tag_embedding_flag:
                    batch_iob_ids[(batch_idx % self.batch_size), i] = iob_tag_id

                if self.position_embedding_flag:
                    batch_distance_protein[(batch_idx % self.batch_size), i] = distance_to_protein_id
                    batch_distance_chemical[(batch_idx % self.batch_size), i] = distance_to_chemical_id

        # obtain sequence lengths
        batch_seq_lens = candidate_relation_dataset['seq_lens'][batch_start_idx:batch_end_idx]

        # obtain labels and convert it to one-hot representation
        batch_labels = candidate_relation_dataset['labels'][batch_start_idx:batch_end_idx]
        batch_labels = self.convert_one_hot(batch_labels)

        # update next batch idx with current batch end idx
        self.dataset[dataset_type]['batch_idx'] = batch_end_idx

        if dataset_type != "training":
            batch_word_embedding_ids = self.trim_batch_data(batch_word_embedding_ids)
            if self.position_embedding_flag:
                batch_distance_protein = self.trim_batch_data(batch_distance_protein)
                batch_distance_chemical = self.trim_batch_data(batch_distance_chemical)
            if self.pos_tag_embedding_flag:
                batch_pos_ids = self.trim_batch_data(batch_pos_ids)
            if self.iob_tag_embedding_flag:
                batch_iob_ids = self.trim_batch_data(batch_iob_ids)

            batch_seq_lens = self.trim_batch_data(batch_seq_lens)

        batch_result = [batch_word_embedding_ids, batch_seq_lens, batch_labels]
        if self.pos_tag_embedding_flag:
            batch_result.append(batch_pos_ids)
        if self.iob_tag_embedding_flag:
            batch_result.append(batch_iob_ids)
        if self.position_embedding_flag:
            batch_result.append(batch_distance_protein)
            batch_result.append(batch_distance_chemical)

        return batch_result

    def trim_batch_data(self, data):
        training_max_length = self.dataset['training']['max_seq_len']
        if isinstance(data, list):
            seq_lens = np.zeros([self.batch_size])
            for i in range(len(data)):
                if data[i] > training_max_length:
                    seq_lens[i] = training_max_length
                else:
                    seq_lens[i] = data[i]
            return seq_lens
        else:  # trim 2d batch data
            current_sequence_length = data.shape[1]
            result = np.zeros((self.batch_size, training_max_length))

            if training_max_length >= current_sequence_length:
                result[:, :current_sequence_length] = data
                return result
            else:
                data = data[:, :training_max_length]
                return data

    def get_embedding_information(self):
        result_dict = {'position_embedding_flag': self.position_embedding_flag,
                       'pos_tag_embedding_flag': self.pos_tag_embedding_flag,
                       'iob_tag_embedding_flag': self.iob_tag_embedding_flag}
        if self.position_embedding_flag:
            result_dict['position_embedding_size'] = self.position_embedding_size
            result_dict['position_ids_size'] = self.self.position_embedding_matrix.shape[0]
        if self.pos_tag_embedding_flag:
            result_dict['pos_tag_embedding_size'] = self.pos_tag_embedding_size
            result_dict['pos_tag_ids_size'] = self.pos_tag_embedding_matrix.shape[0]
        if self.iob_tag_embedding_flag:
            result_dict['iob_tag_embedding_flag'] = self.iob_tag_embedding_size
            result_dict['iob_tag_ids_size'] = self.iob_tag_embedding_matrix.shape[0]

        return result_dict

    def add_false_positive(self, dataset_type, fp_idx):
        dataset = self.dataset[dataset_type]
        batch_start_idx = dataset['batch_idx'] - self.batch_size
        for i in range(len(fp_idx)):
            idx = fp_idx[i]
            idx = batch_start_idx + idx
            dataset['false_positive'].append(idx)

    def add_false_negative(self, dataset_type, fn_idx):
        dataset = self.dataset[dataset_type]
        batch_start_idx = dataset['batch_idx'] - self.batch_size
        for i in range(len(fn_idx)):
            idx = fn_idx[i]
            idx = batch_start_idx + idx
            dataset['false_negative'].append(idx)

    def print_information(self):
        print('Dataset Information:')
        print('Training Set Size: {}'.format(len(self.dataset['training']['data'])))
        print('Development Set Size: {}'.format(len(self.dataset['development']['data'])))
        print('Test Set Size: {}\n'.format(len(self.dataset['test']['data'])))
        print('Data Interface Information:')
        print('Root Directory: {}'.format(self.root_directory))
        print('Dataset Source: {}'.format(self.dataset_source))
        print('Batch Size: {}'.format(self.batch_size))
        print('Text Selection: {}'.format(self.text_selection))
        print('Relation Type: {}'.format(self.relation_type))
        print('Word Tokenizer: {}'.format(self.word_tokenizer))
        print('Word Embedding Directory: {}\n'.format(self.word_embedding_dir))

        print('Position Embedding Information:')
        print('Flag: {}'.format(self.position_embedding_flag))
        if self.position_embedding_flag:
            print('Directory: {}'.format(self.position_embedding_dir))
            print('Size: {}\n'.format(self.position_embedding_size))

        print('POS Tag Embedding Information:')
        print('Flag: {}'.format(self.pos_tag_embedding_flag))
        if self.pos_tag_embedding_flag:
            print('Directory: {}'.format(self.pos_tag_embedding_dir))
            print('Size: {}\n'.format(self.pos_tag_embedding_size))

        print('IOB Tag Embedding Information:')
        print('Flag: {}'.format(self.iob_tag_embedding_flag))
        if self.iob_tag_embedding_flag:
            print('Directory: {}'.format(self.iob_tag_embedding_dir))
            print('Size: {}'.format(self.iob_tag_embedding_size))

    def write_information(self, file):
        file.write('Dataset Information: \n')
        file.write('Training Set Size: {}\n'.format(len(self.dataset['training']['data'])))
        file.write('Development Set Size: {}\n'.format(len(self.dataset['development']['data'])))
        file.write('Test Set Size: {}\n \n'.format(len(self.dataset['test']['data'])))
        file.write('Data Interface Information:\n')
        file.write('Root Directory: {}\n'.format(self.root_directory))
        file.write('Dataset Source: {}\n'.format(self.dataset_source))
        file.write('Batch Size: {}\n'.format(self.batch_size))
        file.write('Text Selection: {}\n'.format(self.text_selection))
        file.write('Relation Type: {}\n'.format(self.relation_type))
        file.write('Word Tokenizer: {}\n'.format(self.word_tokenizer))
        file.write('Word Embedding Directory: {}\n \n'.format(self.word_embedding_dir))

        file.write('Position Embedding Information:\n')
        file.write('Flag: {}\n'.format(self.position_embedding_flag))
        if self.position_embedding_flag:
            file.write('Directory: {}\n'.format(self.position_embedding_dir))
            file.write('Size: {}\n \n'.format(self.position_embedding_size))

        file.write('POS Tag Embedding Information: \n')
        file.write('Flag: {}\n'.format(self.pos_tag_embedding_flag))
        if self.pos_tag_embedding_flag:
            file.write('Directory: {}\n'.format(self.pos_tag_embedding_dir))
            file.write('Size: {}\n \n'.format(self.pos_tag_embedding_size))

        file.write('IOB Tag Embedding Information:\n')
        file.write('Flag: {}\n'.format(self.iob_tag_embedding_flag))
        if self.iob_tag_embedding_flag:
            file.write('Directory: {}\n'.format(self.iob_tag_embedding_dir))
            file.write('Size: {}\n'.format(self.iob_tag_embedding_size))

    @staticmethod
    def convert_one_hot(data):
        result_data = []
        for label in data:
            np_label = np.zeros(2)
            np_label[label] = 1
            result_data.append(np_label)
        return result_data
