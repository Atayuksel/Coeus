import bc_dataset


class DataInterface(object):

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
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

            print('askdjaklsd')

    def parse_dataset(self, data_dictionary, data, labels, full_text, binary_relation):
        for i in range(len(data)):
            instance = data[i]
            label = labels[i]
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

            if not full_text:
                instance_text = self.trim_sentence(instance_text, arg1_text, int(arg1_start_idx),
                                                   arg2_text, int(arg2_start_idx))

            if binary_relation and label != 0:
                label = 1

            data_dictionary['abstract_ids'].append(instance_id)
            data_dictionary['data'].append(instance_text)
            data_dictionary['entities'].append((arg1_text, arg2_text))
            data_dictionary['entity_ids'].append((arg1_id, arg2_id))
            data_dictionary['labels'].append(label)

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
