import progressbar
import nltk
import os
import pickle
from itertools import combinations


class BioCreativeData(object):

    def __init__(self, input_root, output_root, sent_tokenizer, binary_label):
        self.input_root = input_root
        self.output_root = output_root
        self.tokenizer = sent_tokenizer  # 'NLTK' or 'GENIA'
        self.binary_label = binary_label # True or False
        self.dataset = [None] * 6

        # Check for previously created dataset.
        dataset_info_dir = os.path.join(output_root, 'dataset_info.txt')
        dataset_created = False
        if os.path.exists(dataset_info_dir):
            dataset_created = True

        if dataset_created:
            self.dataset[0], self.dataset[1] = self.load_dataset(dataset='TRAINING')
            self.dataset[2], self.dataset[3] = self.load_dataset(dataset='DEV')
            self.dataset[4], self.dataset[5] = self.load_dataset(dataset='TEST')
        else:
            self.dataset[0], self.dataset[1] = self.prepare_dataset(dataset='TRAINING')
            self.dataset[2], self.dataset[3] = self.prepare_dataset(dataset='DEV')
            self.dataset[4], self.dataset[5] = self.prepare_dataset(dataset='TEST')

    def load_dataset(self, dataset):
        """
        Load previously prepared dataset.
        :param dataset: dataset type
        :return data: data list
        :return labels: labels list
        """

        data_directory = ''
        label_directory = ''
        if dataset == 'TRAINING':
            data_directory = os.path.join(self.output_root, 'training_data.pickle')
            label_directory = os.path.join(self.output_root, 'training_label.pickle')
        elif dataset == 'DEV':
            data_directory = os.path.join(self.output_root, 'dev_data.pickle')
            label_directory = os.path.join(self.output_root, 'dev_label.pickle')
        elif dataset == 'TEST':
            data_directory = os.path.join(self.output_root, 'test_data.pickle')
            label_directory = os.path.join(self.output_root, 'test_label.pickle')

        with open(data_directory, 'rb') as file:
            data = pickle.load(file)
        with open(label_directory, 'rb') as file:
            labels = pickle.load(file)
        return data, labels

    def prepare_dataset(self, dataset):
        """
        Derive candidate relations from training data.
        :return list: list contains data, labels, pos mapping, iob mapping, longest sequence and word frequencies.
        TODO: Write example of return list.
        """

        abstracts_location = ''
        entities_location = ''
        relations_location = ''
        if dataset == 'TRAINING':
            root_directory = os.path.join(self.input_root, 'chemprot_training', 'chemprot_training')
            abstracts_location = os.path.join(root_directory, "chemprot_training_abstracts.tsv")
            entities_location = os.path.join(root_directory, "chemprot_training_entities.tsv")
            relations_location = os.path.join(root_directory, "chemprot_training_relations.tsv")
        elif dataset == 'DEV':
            root_directory = os.path.join(self.input_root, 'chemprot_development', 'chemprot_development')
            abstracts_location = os.path.join(root_directory, "chemprot_development_abstracts.tsv")
            entities_location = os.path.join(root_directory, "chemprot_development_entities.tsv")
            relations_location = os.path.join(root_directory, "chemprot_development_relations.tsv")
        elif dataset == 'TEST':
            root_directory = os.path.join(self.input_root, 'chemprot_test_gs', 'chemprot_test_gs')
            abstracts_location = os.path.join(root_directory, "chemprot_test_abstracts_gs.tsv")
            entities_location = os.path.join(root_directory, "chemprot_test_entities_gs.tsv")
            relations_location = os.path.join(root_directory, "chemprot_test_relations_gs.tsv")

        abstracts = self.parse_abstract_file(abstracts_location)
        entities = self.parse_entity_file(entities_location)
        relations = self.parse_relation_file(relations_location)

        data = []
        labels = []

        progress_bar = progressbar.ProgressBar(maxval=len(abstracts),
                                               widgets=[progressbar.Bar('=', '[', ']'),
                                                        ' ',
                                                        progressbar.Percentage()])
        progress_bar_counter = 0
        progress_bar.start()

        for abstract_id, abstract_text in abstracts.items():
            # get sentences
            sentences = nltk.sent_tokenize(abstract_text)

            # get sentence boundaries
            sentences_offsets = []
            for sentence in sentences:
                sentences_offsets.append(abstract_text.find(sentence))
            sentences_offsets.append(len(abstract_text))

            # find entities in the sentences
            sentence_entity_mapping = {}
            entities_in_abstract = entities[abstract_id]
            for entity in entities_in_abstract:
                entity_start_idx = entity['start_idx']
                entity_end_idx = entity['end_idx']
                for sentence_id in range(len(sentences)):
                    sentence_start = sentences_offsets[sentence_id]
                    sentence_end = sentences_offsets[sentence_id + 1]
                    if sentence_start <= int(entity_start_idx) and int(entity_end_idx) <= sentence_end:
                        if sentence_id not in sentence_entity_mapping:
                            sentence_entity_mapping[sentence_id] = []
                        sentence_entity_mapping[sentence_id].append(entity)
                        break

            # create training data
            for sentence_id, entities_in_sentence in sentence_entity_mapping.items():
                sentence_text = sentences[sentence_id]
                comb = combinations(entities_in_sentence, 2)
                for pair in comb:
                    arg1_entity_type = pair[0]['entityType']
                    arg2_entity_type = pair[1]['entityType']

                    if (arg1_entity_type == 'CHEMICAL' and arg2_entity_type != 'CHEMICAL') or \
                            (arg1_entity_type != 'CHEMICAL' and arg2_entity_type == 'CHEMICAL'):
                        sentence_start = sentences_offsets[sentence_id]

                        arg1_start = int(pair[0]['start_idx']) - sentence_start
                        arg2_start = int(pair[1]['start_idx']) - sentence_start

                        instance = [abstract_id, sentence_text,
                                    pair[0]['entityID'], pair[0]['name'], pair[0]['entityType'], str(arg1_start),
                                    pair[1]['entityID'], pair[1]['name'], pair[1]['entityType'], str(arg2_start)]
                        data.append(instance)

                        relations_in_abstracts = []
                        if abstract_id in relations:
                            relations_in_abstracts = relations[abstract_id]

                        label = self.check_instance_relation(relations_in_abstracts,
                                                             instance[2],
                                                             instance[6])
                        label = int(label)
                        labels.append(label)

            progress_bar_counter = progress_bar_counter + 1
            progress_bar.update(progress_bar_counter)
        progress_bar.finish()

        return data, labels

    def write_dataset(self, dataset):
        """
        Write prepared dataset to file
        :param dataset: dataset type
        :return:
        """

        data_directory = ''
        label_directory = ''
        data = []
        labels = []
        if dataset == 'TRAINING':
            data_directory = os.path.join(self.output_root, 'training_data')
            label_directory = os.path.join(self.output_root, 'training_label')
            data = self.dataset[0]
            labels = self.dataset[1]
        elif dataset == 'DEV':
            data_directory = os.path.join(self.output_root, 'dev_data')
            label_directory = os.path.join(self.output_root, 'dev_label')
            data = self.dataset[2]
            labels = self.dataset[3]
        elif dataset == 'TEST':
            data_directory = os.path.join(self.output_root, 'test_data')
            label_directory = os.path.join(self.output_root, 'test_label')
            data = self.dataset[4]
            labels = self.dataset[5]

        with open(data_directory, 'wb') as file:
            pickle.dump(data, file)
        with open(label_directory, 'wb') as file:
            pickle.dump(labels, file)

    @staticmethod
    def check_instance_relation(relation_list, arg1, arg2):
        """
        Takes the candidate relation and assign label to it.
        :param relation_list: relations list in abstract
        :param arg1: entity id of argument 1 in candidate relation
        :param arg2: entity id of argument 2 in candidate relation
        :return: (string) candidate relation label
        """

        for relation in relation_list:
            relation_arg1 = relation['arg_1']
            relation_arg2 = relation['arg_2']
            if relation_arg1 == arg1 and relation_arg2 == arg2:
                return relation['CPR']
            elif relation_arg2 == arg1 and relation_arg1 == arg2:
                return relation['CPR']
        return '0'

    @staticmethod
    def parse_abstract_file(file_location):
        """
        Takes BioCreative abstract file and returns a dictionary.
        :return result: dictionary stores abstract ids and corresponding abstract
        '16357751' = full text of the abstract
        """

        result = {}
        with open(file_location, 'r', encoding="utf8") as file:
            lines = file.readlines()
        for line in lines:
            line = line.split('\t')
            abstract_id = line[0]
            abstract_title = line[1]
            abstract_text = line[2]
            full_abstract_text = abstract_title + ' ' + abstract_text
            result[abstract_id] = full_abstract_text
        return result

    @staticmethod
    def parse_entity_file(file_location):
        """
        takes biocreative entity file and return a dictionary
        :return result: entities dictionary. Each abstract key contains a entity list
        '11319232' = [{'abstractID': '11319232', 'entityID': 'T1', 'entityType': 'CHEMICAL', 'start_idx': '242', 'end_idx': '251', 'name': 'acyl-CoAs'},
                      {'abstractID': '11319232', 'entityID': 'T2', 'entityType': 'CHEMICAL', 'start_idx': '1193', 'end_idx': '1201', 'name': 'triacsin'},
                      ... , ]
        """

        result = {}
        with open(file_location, 'r', encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            line = line.split('\t')
            entity = {'abstractID': line[0],
                      'entityID': line[1],
                      'entityType': line[2],
                      'start_idx': line[3],
                      'end_idx': line[4],
                      'name': line[5][:-1]}
            if line[0] not in result:
                result[line[0]] = []
            result[line[0]].append(entity)
        return result

    @staticmethod
    def parse_relation_file(file_location):
        """
        takes the biocreative relation files and returns content dictionary
        :return result: (dictionary) abstract_id -> (list)relations
        '10047461' = [{'abstractID': '10047461', 'CPR': '3', 'eval_status': 'Y ', 'relation_type': 'ACTIVATOR', 'arg_1': 'T13', 'arg_2': 'T57'},
                      {'abstractID': '10047461', 'CPR': '3', 'eval_status': 'Y ', 'relation_type': 'ACTIVATOR', 'arg_1': 'T7', 'arg_2': 'T39'},
                      ... ]
        """

        result = {}
        with open(file_location, 'r', encoding="utf-8") as file:
            lines = file.readlines()
        for line in lines:
            line = line.split('\t')
            cpr_value = line[1].split(':')[-1]
            arg1_value = line[4].split(':')[-1]
            arg2_value = line[5][:-1].split(':')[-1]
            relation = {'abstractID': line[0],
                        'CPR': cpr_value,
                        'eval_status': line[2],
                        'relation_type': line[3],
                        'arg_1': arg1_value,
                        'arg_2': arg2_value}

            if line[0] not in result:
                result[line[0]] = []
            result[line[0]].append(relation)
        return result
