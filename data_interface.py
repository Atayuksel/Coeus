import bc_dataset
import nltk
import numpy as np


class DataInterface(object):

    def __init__(self, dataset_name, embedding_dir, batch_size, text_selection, binary_relation, pos_embedding):
        self.embedding_dir = embedding_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.pos_embedding_flag = pos_embedding
        self.text_selection = text_selection  # full, part, e_part
        self.binary_relation = binary_relation  # True, False

        self.word_tokenizer = 'NLTK'
        self.embedding_dim = 0
        self.dataset = {'training': {'data': [], 'labels': [], 'entities': [], 'abstract_ids': [], 'entity_ids': [],
                                     'seq_lens': [], 'false_positive': [], 'false_negative': [],
                                     'argument_locations': [], 'max_seq_len': 0, 'batch_idx': 0},
                        'development': {'data': [], 'labels': [], 'entities': [], 'abstract_ids': [], 'entity_ids': [],
                                        'seq_lens': [], 'false_positive': [], 'false_negative': [],
                                        'argument_locations': [], 'max_seq_len': 0, 'batch_idx': 0},
                        'test': {'data': [], 'labels': [], 'entities': [], 'abstract_ids': [], 'entity_ids': [],
                                 'seq_lens': [], 'false_positive': [], 'false_negative': [],
                                 'argument_locations': [], 'max_seq_len': 0, 'batch_idx': 0}}

        if dataset_name == 'BioCreative':
            self.bc_dataset = bc_dataset.BioCreativeData(input_root='dataset',
                                                         output_root='output/bc_dataset',
                                                         sent_tokenizer='NLTK',
                                                         binary_label=True)

            dataset_validity = self.check_dataset(self.bc_dataset)
            self.raw_dataset = self.bc_dataset.dataset
            if dataset_validity:
                self.parse_dataset(self.dataset['training'], self.raw_dataset[0], self.raw_dataset[1],
                                   self.text_selection, self.binary_relation)
                self.parse_dataset(self.dataset['development'], self.raw_dataset[2], self.raw_dataset[3],
                                   self.text_selection, self.binary_relation)
                self.parse_dataset(self.dataset['test'], self.raw_dataset[4], self.raw_dataset[5],
                                   self.text_selection, self.binary_relation)

            self.pos_to_id = self.create_pos_embeddings()
            self.embeddings, self.word_to_id = self.parse_embedding()

    def parse_embedding(self):
        embedding_file = open(self.embedding_dir, 'r', encoding='utf-8')
        lines = embedding_file.readlines()
        embedding_file.close()

        word_id_mapping = {'unk': 0, 'pad':1}
        vocab_size = len(lines)
        self.embedding_dim = len(lines[0][:-1].split(' ')) - 1
        embeddings = np.zeros(((vocab_size+2), self.embedding_dim))
        embeddings[0:1, :] = np.random.rand(1, self.embedding_dim)

        for idx in range(len(lines)):
            line = lines[idx][:-1].split(' ')
            token = line[0]

            # get embedding and convert it to numpy array
            word_embedding = line[1:]
            word_embedding = list(np.float_(word_embedding))
            word_embedding = np.asarray(word_embedding)

            # add embedding to embeddings
            embeddings[idx+2, :] = word_embedding

            # assign id to token
            current_id = len(word_id_mapping)
            word_id_mapping[token] = current_id

        return embeddings, word_id_mapping

    def parse_dataset(self, data_dictionary, data, labels, text_selection, binary_relation):
        for i in range(len(data)):
            label = labels[i]

            instance = data[i]
            instance_id = instance[0]
            instance_text = instance[1]

            arg1_id = instance[2]
            arg1_text = instance[3]
            arg1_type = instance[4]
            arg1_start_idx = instance[5]

            arg2_id = instance[6]
            arg2_text = instance[7]
            arg2_type = instance[8]
            arg2_start_idx = instance[9]

            # full text sentence or text between entities
            protein_loc, chemical_loc, tokenized_sentence = self.preprocess_sentence(sentence=instance_text,
                                                                                     arg1_text=arg1_text,
                                                                                     arg1_start=int(arg1_start_idx),
                                                                                     arg1_type=arg1_type,
                                                                                     arg2_text=arg2_text,
                                                                                     arg2_start=int(arg2_start_idx),
                                                                                     arg2_type=arg2_type,
                                                                                     sentence_trim=text_selection)
            # binary relation or multiclass relation
            if binary_relation and label != 0:
                label = 1

            # check for sentence empty or not
            if len(instance_text) != 0:
                # get maximum sequence length
                if data_dictionary['max_seq_len'] < len(tokenized_sentence):
                    data_dictionary['max_seq_len'] = len(tokenized_sentence)

                data_dictionary['abstract_ids'].append(instance_id)
                data_dictionary['data'].append(tokenized_sentence)
                data_dictionary['entities'].append((arg1_text, arg2_text))
                data_dictionary['entity_ids'].append((arg1_id, arg2_id))
                data_dictionary['labels'].append(label)
                data_dictionary['seq_lens'].append(len(tokenized_sentence))
                data_dictionary['argument_locations'].append((protein_loc, chemical_loc))

    def get_batch(self, dataset_type):
        dataset = self.dataset[dataset_type]
        if dataset['batch_idx'] == len(dataset['data']):
            dataset['batch_idx'] = 0

        batch_start_idx = dataset['batch_idx']
        batch_end_idx = min([batch_start_idx + self.batch_size, len(dataset['data'])])

        # get batch data
        pad_id_data = self.word_to_id['pad']
        pad_id_distance = self.pos_to_id['pad']

        batch_data = np.full((self.batch_size, dataset['max_seq_len']), pad_id_data)
        batch_distance_protein = np.full((self.batch_size, dataset['max_seq_len']), pad_id_distance)
        batch_distance_chemical = np.full((self.batch_size, dataset['max_seq_len']), pad_id_distance)

        for batch_idx in range(batch_start_idx, batch_end_idx):
            tokenized_text = dataset['data'][batch_idx]
            protein_loc = dataset['argument_locations'][batch_idx][0]
            chemical_loc = dataset['argument_locations'][batch_idx][1]
            for token_idx in range(len(tokenized_text)):
                token = tokenized_text[token_idx]
                distance_to_protein = token_idx - protein_loc
                distance_to_chemical = token_idx - chemical_loc
                if token in self.word_to_id:
                    token_id = self.word_to_id[token]
                else:
                    token_id = 0

                if distance_to_protein >= self.dataset['training']['max_seq_len']:
                    distance_to_protein = self.dataset['training']['max_seq_len'] - 1
                elif distance_to_protein <= -self.dataset['training']['max_seq_len']:
                    distance_to_protein = -self.dataset['training']['max_seq_len'] + 1

                if distance_to_chemical >= self.dataset['training']['max_seq_len']:
                    distance_to_chemical = self.dataset['training']['max_seq_len'] + 1
                elif distance_to_chemical <= -self.dataset['training']['max_seq_len']:
                    distance_to_chemical = -self.dataset['training']['max_seq_len'] + 1

                distance_to_protein_id = self.pos_to_id[distance_to_protein]
                distance_to_chemical_id = self.pos_to_id[distance_to_chemical]

                batch_data[(batch_idx % self.batch_size), token_idx] = token_id
                batch_distance_protein[(batch_idx % self.batch_size), token_idx] = distance_to_protein_id
                batch_distance_chemical[(batch_idx % self.batch_size), token_idx] = distance_to_chemical_id

        # get batch sequence length
        batch_seq_lens = dataset['seq_lens'][batch_start_idx:batch_end_idx]

        # get batch one hot labels
        batch_labels = dataset['labels'][batch_start_idx:batch_end_idx]
        batch_labels = self.convert_one_hot(batch_labels)

        self.dataset[dataset_type]['batch_idx'] = batch_end_idx

        if dataset_type != "training":
            batch_data = self.postprocess_data(batch_data)
            batch_distance_protein = self.postprocess_data(batch_distance_protein)
            batch_distance_chemical = self.postprocess_data(batch_distance_chemical)
            self.process_seq_length(batch_seq_lens)

        return batch_data, batch_labels, batch_seq_lens, batch_distance_protein, batch_distance_chemical

    def postprocess_data(self, data):
        training_max_length = self.dataset['training']['max_seq_len']
        current_max_length = data.shape[1]
        result_data = np.zeros((self.batch_size, training_max_length))

        if training_max_length >= current_max_length:  # pad dataset
            result_data[:, :current_max_length] = data
            return result_data
        else:  # crop dataset
            data = data[:, :training_max_length]
            return data

    def process_seq_length(self, seq_lens):
        limit_max_length = self.dataset['training']['max_seq_len']
        for idx in range(len(seq_lens)):
            if seq_lens[idx] > limit_max_length:
                seq_lens[idx] = limit_max_length

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

    def write_results(self, dataset_type):
        dataset = self.dataset[dataset_type]
        false_positive = dataset['false_positive']
        false_negative = dataset['false_negative']
        with open("false_positive.txt", 'w') as file:
            for idx in false_positive:
                data = dataset['data'][idx]
                raw_instance = self.raw_dataset[2][idx]
                file.write('{}\nSelected Text: {}\n'.format(idx, ' '.join(data)))
                file.write('Full Text: {}\nArgument1:{}, {}, {}\nArgument2:{}, {}, {}\n\n'.format(raw_instance[1],
                                                                                                  raw_instance[2],
                                                                                                  raw_instance[3],
                                                                                                  raw_instance[4],
                                                                                                  raw_instance[6],
                                                                                                  raw_instance[7],
                                                                                                  raw_instance[8])
                           )
        with open("false_negative.txt", 'w') as file:
            for idx in false_negative:
                data = dataset['data'][idx]
                raw_instance = self.raw_dataset[2][idx]
                file.write('{}\nSelected Text: {}\n'.format(idx, ' '.join(data)))
                file.write('Full Text: {}\nArgument1:{}, {}, {}\nArgument2:{}, {}, {}\n\n'.format(raw_instance[1],
                                                                                                  raw_instance[2],
                                                                                                  raw_instance[3],
                                                                                                  raw_instance[4],
                                                                                                  raw_instance[6],
                                                                                                  raw_instance[7],
                                                                                                  raw_instance[8])
                           )

    def create_pos_embeddings(self):
        pos_embedding_size = 50
        dataset = self.dataset['training']
        training_max_seq_len = dataset['max_seq_len']
        pos_to_id = {0: 0, 'pad':1}
        for i in range(1, training_max_seq_len):
            pos_to_id[i] = len(pos_to_id)
        for i in range(1, training_max_seq_len):
            pos_to_id[-i] = len(pos_to_id)
        return pos_to_id

    @staticmethod
    def preprocess_sentence(sentence, arg1_text, arg1_start, arg1_type,
                            arg2_text, arg2_start, arg2_type, sentence_trim):
        if arg1_start < arg2_start:
            trim_start_idx = arg1_start
            trim_end_idx = arg2_start + len(arg2_text)
            instance_text = sentence[trim_start_idx:trim_end_idx]
            if arg1_type == 'CHEMICAL':
                first_entity_type = 'chemical'
            else:
                first_entity_type = 'gene'
        else:
            trim_start_idx = arg2_start
            trim_end_idx = arg1_start + len(arg1_text)
            instance_text = sentence[trim_start_idx:trim_end_idx]
            if arg2_type == 'CHEMICAL':
                first_entity_type = 'chemical'
            else:
                first_entity_type = 'gene'
        trim = instance_text.strip()
        if sentence_trim == 'part':
            tokenized_sent = nltk.word_tokenize(trim)
            if first_entity_type == 'chemical':
                chemical_location = 0
                protein_location = len(tokenized_sent) - 1
            else:
                chemical_location = len(tokenized_sent) - 1
                protein_location = 0
            return protein_location, chemical_location, tokenized_sent
        elif sentence_trim == 'e_part':
            sentence_begin = sentence[0:trim_start_idx]
            sentence_end = sentence[trim_end_idx:len(sentence)]
            sentence_begin = nltk.word_tokenize(sentence_begin)
            sentence_end = nltk.word_tokenize(sentence_end)
            if len(sentence_begin) > 1:
                extended_trim = sentence_begin[-2] + " " + sentence_begin[-1] + " " + extended_trim
            elif len(sentence_begin) == 1:
                extended_trim = sentence_begin[-1] + " " + extended_trim
            if len(sentence_end) > 1:
                extended_trim = extended_trim + " " + sentence_end[0] + " " + sentence_end[1]
            elif len(sentence_end) == 1:
                extended_trim = extended_trim + " " + sentence_end[0]

            if first_entity_type == 'chemical':
                chemical_location = 2
                protein_location = len(extended_trim) - 3
            else:
                chemical_location = len(extended_trim) - 3
                protein_location = 2
            return protein_location, chemical_location, extended_trim
        elif sentence_trim == 'full':
            sentence_begin = sentence[0:trim_start_idx]
            sentence_end = sentence[trim_end_idx:len(sentence)]
            tokenized_sentence_begin = nltk.word_tokenize(sentence_begin)
            tokenized_sentence_end = nltk.word_tokenize(sentence_end)
            tokenized_sentence_full = nltk.word_tokenize(sentence)
            if first_entity_type == 'chemical':
                chemical_location = len(tokenized_sentence_begin)
                protein_location = len(tokenized_sentence_full) - len(tokenized_sentence_end) - 1
            else:
                chemical_location = len(tokenized_sentence_full) - len(tokenized_sentence_end) - 1
                protein_location = len(tokenized_sentence_begin)
            return protein_location, chemical_location, tokenized_sentence_full

    @staticmethod
    def check_dataset(dataset):
        """
        Check for dataset object shape and style.
        :return check_flag: True if dataset is valid and False if dataset is invalid.
        """

        if hasattr(dataset, 'dataset'):  # biocreative dataset object has a list titled 'dataset'
            if len(dataset.dataset) == 6:  # 'dataset' list should contain 6 items.
                for i in range(3):
                    start_idx = i * 2
                    data = dataset.dataset[start_idx]
                    label = dataset.dataset[start_idx+1]
                    if len(data) == len(label):  # data and label should contain same number of instance
                        for j in range(len(data)):
                            if len(data[j]) != 10 or not (isinstance(label[j], int)):
                                return False
                    else:
                        return False
            else:
                return False
        else:
            return False
        return True

    @staticmethod
    def convert_one_hot(data):
        result_data = []
        for label in data:
            np_label = np.zeros(2)
            np_label[label] = 1
            result_data.append(np_label)
        return result_data
