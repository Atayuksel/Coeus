import bc_dataset


class DataInterface(object):

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        # self.training_data = []
        # self.training_labels = []
        # self.training_entities = []
        # self.training_abstract_ids = []
        # self.training_entity_ids = []

        self.dataset = {'training': {'data':[], 'labels':[], 'entities':[], 'abstract_ids':[], 'entity_ids':[]},
                        'development': {'data':[], 'labels':[], 'entities':[], 'abstract_ids':[], 'entity_ids':[]},
                        'test': {'data':[], 'labels':[], 'entities':[], 'abstract_ids':[], 'entity_ids':[]}}

        if dataset_name == 'BioCreative':
            self.bc_dataset = bc_dataset.BioCreativeData(input_root='dataset',
                                                         output_root='output/bc_dataset',
                                                         tokenizer='NLTK',
                                                         output_style='FULL',
                                                         binary_label=True)
            dataset_validity = self.check_dataset(self.bc_dataset)
            self.raw_dataset = self.bc_dataset.dataset
            if dataset_validity:
                self.parse_dataset(self.dataset['training'], self.raw_dataset[0], self.raw_dataset[1], False)

            print('askdjaklsd')

    def check_dataset(self, dataset):
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
                            if len(data[j]) != 8 or not (isinstance(label[j], int)):
                                return False
                    else:
                        return False
            else:
                return False
        else:
            return False
        return True

    def parse_dataset(self, data_dictionary, data, labels, full_text):
        for i in range(len(data)):
            instance = data[i]
            label = labels[i]
            instance_id = instance[0]
            instance_text = instance[1]
            arg1_id = instance[2]
            arg1_text = instance[3]
            arg1_type = instance[4]
            arg2_id = instance[5]
            arg2_text = instance[6]
            arg2_type = instance[7]

            if not full_text:
                arg1_start_index = instance_text.find(arg1_text)
                arg1_end_index = arg1_start_index + len(arg1_text)
                arg2_start_index = instance_text.find(arg2_text)
                arg2_end_index = arg2_start_index + len(arg2_text)
                if arg1_start_index < arg2_start_index:
                    instance_text = instance_text[arg1_end_index:arg2_start_index]
                else:
                    instance_text = instance_text[arg2_end_index:arg1_start_index]
            instance_text = instance_text.strip()

            data_dictionary['abstract_ids'].append(instance_id)
            data_dictionary['data'].append(instance_text)
            data_dictionary['entities'].append((arg1_text, arg2_text))
            data_dictionary['entity_ids'].append((arg1_id, arg2_id))
            data_dictionary['labels'].append(label)
