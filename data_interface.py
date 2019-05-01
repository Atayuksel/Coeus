import bc_dataset
import nltk
import numpy as np


class DataInterface(object):

    def __init__(self, dataset_name, embedding_dir):
        self.embedding_dir = embedding_dir
        self.dataset_name = dataset_name
        self.word_tokenizer = 'NLTK'
        self.max_seq_length = 0

        self.batch_size = 10
        self.cur_index = 0

        self.embedding_dim = 0

        self.dataset = {'training': {'data': [], 'labels': [], 'entities': [], 'abstract_ids': [], 'entity_ids': []},
                        'development': {'data': [], 'labels': [], 'entities': [], 'abstract_ids': [], 'entity_ids': []},
                        'test': {'data': [], 'labels': [], 'entities': [], 'abstract_ids': [], 'entity_ids': []}}

        if dataset_name == 'BioCreative':
            self.bc_dataset = bc_dataset.BioCreativeData(input_root='dataset',
                                                         output_root='output/bc_dataset',
                                                         sent_tokenizer='NLTK',
                                                         binary_label=True)

            dataset_validity = self.check_dataset(self.bc_dataset)

            self.raw_dataset = self.bc_dataset.dataset
            if dataset_validity:
                self.parse_dataset(self.dataset['training'], self.raw_dataset[0], self.raw_dataset[1], False, True)
                self.parse_dataset(self.dataset['development'], self.raw_dataset[2], self.raw_dataset[3], False, True)
                self.parse_dataset(self.dataset['test'], self.raw_dataset[4], self.raw_dataset[5], False, True)

            self.embeddings, self.word_to_id = self.parse_embedding()

    def parse_embedding(self):
        embedding_file = open(self.embedding_dir, 'r', encoding='utf-8')
        lines = embedding_file.readlines()
        embedding_file.close()

        word_id_mapping = {'unk': 0}
        vocab_size = len(lines)
        self.embedding_dim = len(lines[0][:-1].split(' ')) - 1
        embeddings = np.zeros(((vocab_size+1), self.embedding_dim))
        embeddings[0, :] = np.random.rand(1, self.embedding_dim)

        for idx in range(len(lines)):
            line = lines[idx][:-1].split(' ')
            token = line[0]

            # get embedding and convert it to numpy array
            word_embedding = line[1:]
            word_embedding = list(np.float_(word_embedding))
            word_embedding = np.asarray(word_embedding)

            # add embedding to embeddings
            embeddings[idx+1, :] = word_embedding

            # assign id to token
            current_id = len(word_id_mapping)
            word_id_mapping[token] = current_id

        return embeddings, word_id_mapping

    def parse_dataset(self, data_dictionary, data, labels, full_text, binary_relation):
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
            if not full_text:
                instance_text = self.trim_sentence(instance_text, arg1_text, int(arg1_start_idx),
                                                   arg2_text, int(arg2_start_idx))

            # binary relation or multiclass relation
            if binary_relation and label != 0:
                label = 1

            # check for sentence empty or not
            if len(instance_text) != 0:
                tokenized_sent = nltk.word_tokenize(instance_text)

                # get maximum sequence length
                if self.max_seq_length < len(tokenized_sent):
                    self.max_seq_length = len(tokenized_sent)

                data_dictionary['abstract_ids'].append(instance_id)
                data_dictionary['data'].append(tokenized_sent)
                data_dictionary['entities'].append((arg1_text, arg2_text))
                data_dictionary['entity_ids'].append((arg1_id, arg2_id))
                data_dictionary['labels'].append(label)

    def create_batch(self, dataset_type):
        dataset = self.dataset[dataset_type]
        batch_data = np.zeros((self.batch_size, self.max_seq_length, self.embedding_dim))

        for batch_idx in range(self.cur_index, (self.cur_index+self.batch_size)):
            tokenized_text = dataset['data'][batch_idx]
            for token_idx in range(len(tokenized_text)):
                token = tokenized_text[token_idx]
                if token in self.word_to_id:
                    token_id = self.word_to_id[token]
                else:
                    token_id = 0
                token_embedding = self.embeddings[token_id, :]
                batch_data[batch_idx, token_idx, :] = token_embedding
            self.cur_index += 1

        return batch_data

    @staticmethod
    def trim_sentence(sentence, arg1_text, arg1_start, arg2_text, arg2_start):
        arg1_start
        if arg1_start < arg2_start:
            trim_start_idx = arg1_start + len(arg1_text)
            trim_end_idx = arg2_start
            instance_text = sentence[trim_start_idx:trim_end_idx]
        else:
            trim_start_idx = arg2_start + len(arg2_text)
            trim_end_idx = arg1_start
            instance_text = sentence[trim_start_idx:trim_end_idx]

        trim = instance_text.strip()
        return trim

    @staticmethod
    def check_dataset(dataset):
        """
        Check for dataset object shape and style.
        :return check_flag: True if dataset is valid and False if dataset is invalid.
        """

        if hasattr(dataset, 'dataset'):
            if len(dataset.dataset) == 6:
                for i in range(3):
                    start_idx = i * 2
                    data = dataset.dataset[start_idx]
                    label = dataset.dataset[start_idx+1]
                    if len(data) == len(label):
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
