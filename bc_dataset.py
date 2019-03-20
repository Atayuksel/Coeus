import progressbar
import nltk
import os
from itertools import combinations


class BioCreativeData(object):

    def __init__(self, dataset_location):
        self.dataset_location = dataset_location

        self.training = {}
        self.development = {}
        self.test = {}

        result = self.prepare_training_dataset()
        self.training["data"] = result[0]
        self.training["label"] = result[1]
        self.word_frequencies = result[2]
        self.pos_tag_mapping = result[3]
        self.iob_tag_mapping = result[4]
        self.max_seq_len = result[5]

        self.development["data"], self.development["label"] = self.prepare_development_dataset()
        self.test["data"], self.test["label"] = self.prepare_test_dataset()

    def prepare_training_dataset(self):
        """
        Create training dataset from BioCreative files.
        :return list: list contains data, labels, pos mapping, iob mapping, longest sequence and word frequencies.
        """

        training_directory = os.path.join(self.dataset_location, 'chemprot_training', 'chemprot_training')

        abstracts_location = os.path.join(training_directory, "chemprot_training_abstracts.tsv")
        entities_location = os.path.join(training_directory, "chemprot_training_entities.tsv")
        relations_location = os.path.join(training_directory, "chemprot_training_relations.tsv")

        abstracts = self.parse_abstract_file(abstracts_location)
        entities = self.parse_entity_file(entities_location)
        relations = self.parse_relation_file(relations_location)

        data = []
        labels = []

        pos_tag_mapping = {'unk': 0}
        iob_tag_mapping = {'unk': 0}
        word_frequencies = {}
        max_seq_length = 0

        progress_bar = progressbar.ProgressBar(maxval=len(abstracts),
                                               widgets=[progressbar.Bar('=', '[', ']'),
                                                        ' ',
                                                        progressbar.Percentage()])

        progress_bar_counter = 0
        progress_bar.start()

        for abstract_id, abstract_text in abstracts.items():
            # get sentences
            sentences = nltk.sent_tokenize(abstract_text)
            sentences_offsets = []

            # for each sentence
            # find sentence indexes
            # update mapping(iob, pos, word_frequency) dictionary
            # find max length sentence
            for sentence in sentences:
                # add sentence index to sentence offsets
                sentences_offsets.append(abstract_text.find(sentence))

                mappings = self.update_id_mappings(sentence=sentence,
                                                   pos_tag_mapping=pos_tag_mapping,
                                                   iob_tag_mapping=iob_tag_mapping,
                                                   word_frequencies=word_frequencies,
                                                   max_seq_length=max_seq_length)
                pos_tag_mapping = mappings[0]
                iob_tag_mapping = mappings[1]
                word_frequencies = mappings[2]
                max_seq_length = mappings[3]

            sentences_offsets.append(len(abstract_text))

            # find entities in the abstract
            # and map each entity to corresponding sentence
            entities_in_abstract = entities[abstract_id]
            sentence_entity_mapping = {}
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

            # for each sentence that contains more than two entity
            # create an instance of candidate relation
            # assign the relation to a sentence
            for sentence_id, entities_in_sentence in sentence_entity_mapping.items():
                sentence_text = sentences[sentence_id]

                comb = combinations(entities_in_sentence, 2)
                for pair in comb:

                    arg1_entity_type = pair[0]['entityType']
                    arg2_entity_type = pair[1]['entityType']

                    if (arg1_entity_type == 'CHEMICAL' and arg2_entity_type != 'CHEMICAL') or \
                            (arg1_entity_type != 'CHEMICAL' and arg2_entity_type == 'CHEMICAL'):

                        instance = [abstract_id, sentence_text,
                                    pair[0]['entityID'], pair[0]['name'], pair[0]['entityType'],
                                    pair[1]['entityID'], pair[1]['name'], pair[1]['entityType']]
                        data.append(instance)

                        relations_in_abstracts = []
                        if abstract_id in relations:
                            relations_in_abstracts = relations[abstract_id]

                        label = self.check_instance_relation(relations_in_abstracts,
                                                             instance[2],
                                                             instance[5])
                        label = int(label)
                        labels.append(label)

            progress_bar_counter = progress_bar_counter + 1
            progress_bar.update(progress_bar_counter)
        progress_bar.finish()

        return [data, labels, word_frequencies, pos_tag_mapping, iob_tag_mapping, max_seq_length]

    def prepare_development_dataset(self):
        """
        prepares development dataset and returns data and label python lists.
        :return data: python list contains data instances
        :return label: python list contains label of corresponding data instances.
        """

        development_directory = os.path.join(self.dataset_location, 'chemprot_development', 'chemprot_development')

        abstracts_location = os.path.join(development_directory, "chemprot_development_abstracts.tsv")
        entities_location = os.path.join(development_directory, "chemprot_development_entities.tsv")
        relations_location = os.path.join(development_directory, "chemprot_development_relations.tsv")

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
            sentences_offsets = []

            # for each sentence
            # find sentence indexes
            for sentence in sentences:
                sentences_offsets.append(abstract_text.find(sentence))
            sentences_offsets.append(len(abstract_text))

            # find entities in the abstract
            # and map each entity to corresponding sentence
            entities_in_abstract = entities[abstract_id]
            sentence_entity_mapping = {}
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

            # for each sentence that contains more than two entity
            # create an instance of candidate relation
            # assign the relation to a sentence
            for sentence_id, entities_in_sentence in sentence_entity_mapping.items():
                sentence_text = sentences[sentence_id]

                comb = combinations(entities_in_sentence, 2)
                for pair in comb:

                    arg1_entity_type = pair[0]['entityType']
                    arg2_entity_type = pair[1]['entityType']

                    if (arg1_entity_type == 'CHEMICAL' and arg2_entity_type != 'CHEMICAL') or \
                            (arg1_entity_type != 'CHEMICAL' and arg2_entity_type == 'CHEMICAL'):

                        instance = [abstract_id, sentence_text,
                                    pair[0]['entityID'], pair[0]['name'], pair[0]['entityType'],
                                    pair[1]['entityID'], pair[1]['name'], pair[1]['entityType']]
                        data.append(instance)

                        relations_in_abstracts = []
                        if abstract_id in relations:
                            relations_in_abstracts = relations[abstract_id]

                        label = self.check_instance_relation(relations_in_abstracts,
                                                             instance[2],
                                                             instance[5])
                        label = int(label)
                        labels.append(label)

            progress_bar_counter = progress_bar_counter + 1
            progress_bar.update(progress_bar_counter)
        progress_bar.finish()

        return data, labels

    def prepare_test_dataset(self):
        """
        prepares test dataset and returns data and label python lists.
        :return data: python list contains data instances
        :return label: python list contains label of corresponding data instances.
        """

        test_directory = os.path.join(self.dataset_location, 'chemprot_test_gs', 'chemprot_test_gs')

        abstracts_location = os.path.join(test_directory, "chemprot_test_abstracts_gs.tsv")
        entities_location = os.path.join(test_directory, "chemprot_test_entities_gs.tsv")
        relations_location = os.path.join(test_directory, "chemprot_test_relations_gs.tsv")

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
            sentences_offsets = []

            # for each sentence
            # find sentence indexes
            for sentence in sentences:
                sentences_offsets.append(abstract_text.find(sentence))
            sentences_offsets.append(len(abstract_text))

            # find entities in the abstract
            # and map each entity to corresponding sentence
            entities_in_abstract = entities[abstract_id]
            sentence_entity_mapping = {}
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

            # for each sentence that contains more than two entity
            # create an instance of candidate relation
            # assign the relation to a sentence
            for sentence_id, entities_in_sentence in sentence_entity_mapping.items():
                sentence_text = sentences[sentence_id]

                comb = combinations(entities_in_sentence, 2)
                for pair in comb:

                    arg1_entity_type = pair[0]['entityType']
                    arg2_entity_type = pair[1]['entityType']

                    if (arg1_entity_type == 'CHEMICAL' and arg2_entity_type != 'CHEMICAL') or \
                            (arg1_entity_type != 'CHEMICAL' and arg2_entity_type == 'CHEMICAL'):

                        instance = [abstract_id, sentence_text,
                                    pair[0]['entityID'], pair[0]['name'], pair[0]['entityType'],
                                    pair[1]['entityID'], pair[1]['name'], pair[1]['entityType']]
                        data.append(instance)

                        relations_in_abstracts = []
                        if abstract_id in relations:
                            relations_in_abstracts = relations[abstract_id]

                        label = self.check_instance_relation(relations_in_abstracts,
                                                             instance[2],
                                                             instance[5])
                        label = int(label)
                        labels.append(label)

            progress_bar_counter = progress_bar_counter + 1
            progress_bar.update(progress_bar_counter)
        progress_bar.finish()

        return data, labels

    @staticmethod
    def update_id_mappings(sentence, pos_tag_mapping, iob_tag_mapping, word_frequencies, max_seq_length):
        """
        update mapping dictionary based on the sentence in parameters
        :param sentence: (string) sentence
        :param pos_tag_mapping: (dictionary) mapping of pos tags to ids
        :param iob_tag_mapping: (dictionary) mapping of iob tags to ids
        :param word_frequencies: (dictionary)
        :param max_seq_length:  (int)
        :return: pos_tag_mapping, iob_tag_mapping, word_frequencies, max_seq_length
        """

        # obtain pos and iob tags of the sentence
        tokens = nltk.word_tokenize(sentence.lower())
        pos_tags = nltk.pos_tag(tokens)
        chunk_tree = nltk.ne_chunk(pos_tags)
        iob_tags = nltk.tree2conlltags(chunk_tree)

        # check for maximum length sequence
        if len(iob_tags) > max_seq_length:
            max_seq_length = len(iob_tags)

        # for each token in the sentence
        # check the token is assigned to a id
        # update word frequency dictionary
        for token in iob_tags:
            token_text = token[0]

            # update word frequency dictionary
            if token_text not in word_frequencies:
                word_frequencies[token_text] = 0
            word_frequencies[token_text] = word_frequencies[token_text] + 1

            # update pos tag mapping
            token_pos_tag = token[1]
            if token_pos_tag not in pos_tag_mapping:
                cur_available_pos_id = len(pos_tag_mapping)
                pos_tag_mapping[token_pos_tag] = cur_available_pos_id

            # update iob tag mapping
            token_iob_tag = token[2]
            if token_iob_tag not in iob_tag_mapping:
                cur_available_iob_id = len(iob_tag_mapping)
                iob_tag_mapping[token_iob_tag] = cur_available_iob_id

        return [pos_tag_mapping, iob_tag_mapping, word_frequencies, max_seq_length]

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
