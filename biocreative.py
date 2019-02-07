import progressbar
import nltk
from itertools import combinations


class BioCreativeData(object):

    def __init__(self, training_abstract_loc, training_entities_loc, training_relations_loc,
                 dev_abstract_loc, dev_entities_loc, dev_relations_loc,
                 test_abstract_loc, test_entities_loc, test_relations_loc):

        self.training_abstract_location = training_abstract_loc
        self.training_entities_location = training_entities_loc
        self.training_relations_location = training_relations_loc

        self.development_abstract_location = dev_abstract_loc
        self.development_entities_location = dev_entities_loc
        self.development_relations_location = dev_relations_loc

        self.test_abstract_location = test_abstract_loc
        self.test_entities_location = test_entities_loc
        self.test_relation_location = test_relations_loc

        self.training_abstracts = {}
        self.training_entities = {}
        self.training_relations = {}

        self.development_abstracts = {}
        self.development_entities = {}
        self.development_relations = {}

        self.test_abstracts = {}
        self.test_entities = {}
        self.test_relations = {}

    def create_biocreative_dataset(self, data=0):
        """
        creates dataset for machine learning models from BioCreative abstract, entities and relations file
        :return:
        inputs: (list)
        [abstract_id, sentence_text, entity_id, entity_name, entity_type, entity_id, entity_name, entity_type]
        labels: (list) int values of labels
        word_freq: (dictionary) word frequencies
        pos_tag_mapping: (dictionary) mapping pos tags to ids
        iob_tag_mapping: (dictionary) mapping iob tags to ids
        max_seq_length: (int) maximum sentence length in the dataset
        """

        if data == 0:
            abstracts_dict = self.training_abstracts = self.parse_abstract_file(self.training_abstract_location)
            entities_dict = self.training_entities = self.parse_entity_file(self.training_entities_location)
            relations_dict = self.training_relations = self.parse_relation_file(self.training_relations_location)
        elif data == 1:
            abstracts_dict = self.development_abstracts = self.parse_abstract_file(self.development_abstract_location)
            entities_dict = self.development_entities = self.parse_entity_file(self.development_entities_location)
            relations_dict = self.development_relations = self.parse_relation_file(self.development_relations_location)
        else:
            abstracts_dict = self.test_abstracts = self.parse_abstract_file(self.test_abstract_location)
            entities_dict = self.test_entities = self.parse_entity_file(self.test_entities_location)
            relations_dict = self.test_relations = self.parse_relation_file(self.test_relation_location)

        inputs = []
        labels = []

        pos_tag_mapping = {'unk': 0}
        iob_tag_mapping = {'unk': 0}
        word_freq = {}
        max_seq_length = 0

        progress_bar = progressbar.ProgressBar(maxval=len(abstracts_dict),
                                               widgets=[progressbar.Bar('=', '[', ']'),
                                                        ' ',
                                                        progressbar.Percentage()])
        progress_bar_counter = 0
        progress_bar.start()

        for abstract_id, abstract_text in abstracts_dict.items():
            # get sentences
            sentences = nltk.sent_tokenize(abstract_text)
            sentence_offsets = []

            # for each sentence
            # find sentence indexes
            # update mapping(iob, pos, word frequency) dictionary
            # find max length sentence
            for sentence in sentences:

                # add sentence index to sentence_offsets
                sentence_offsets.append(abstract_text.find(sentence))

                # if mode is selected training then update mappings and select max sequence length
                if data == 0:
                    mappings = self.update_id_mappings(sentence=sentence,
                                                       pos_tag_mapping=pos_tag_mapping,
                                                       iob_tag_mapping=iob_tag_mapping,
                                                       word_frequencies=word_freq,
                                                       max_seq_length=max_seq_length)
                    pos_tag_mapping = mappings[0]
                    iob_tag_mapping = mappings[1]
                    word_freq = mappings[2]
                    max_seq_length = mappings[3]

            sentence_offsets.append(len(abstract_text))

            # find entities in the abstract
            # and map each entity to corresponding sentence
            entities_in_abstract = entities_dict[abstract_id]
            sentence_entity_mapping = {}
            for entity in entities_in_abstract:
                entity_start_idx = entity['start_idx']
                entity_end_idx = entity['end_idx']
                for sentence_id in range(len(sentences)):
                    sentence_start = sentence_offsets[sentence_id]
                    sentence_end = sentence_offsets[sentence_id+1]
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
                        inputs.append(instance)

                        relations_in_abstracts = []
                        if abstract_id in relations_dict:
                            relations_in_abstracts = relations_dict[abstract_id]

                        label = self.check_instance_relation(relations_in_abstracts,
                                                             instance[2],
                                                             instance[5])
                        label = int(label)
                        labels.append(label)

            progress_bar_counter = progress_bar_counter + 1
            progress_bar.update(progress_bar_counter)

        progress_bar.finish()

        if data == 0:
            return inputs, labels, word_freq, pos_tag_mapping, iob_tag_mapping, max_seq_length
        else:
            return inputs, labels

    @staticmethod
    def update_id_mappings(sentence, pos_tag_mapping, iob_tag_mapping, word_frequencies, max_seq_length):
        """
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
        takes the candidate relation and assign label to it.
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
        Takes BioCreative Abstract file and returns a dictionary.
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
