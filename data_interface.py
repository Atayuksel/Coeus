import bc_dataset


class DataInterface(object):

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        self.training_data = []
        self.training_labels = []
        self.training_entities = []
        self.training_abstract_ids = []
        self.training_entity_ids = []

        if dataset_name == 'BioCreative':
            self.biocreative_dataset = bc_dataset.BioCreativeData(input_root='dataset',
                                                             output_root='output/bc_dataset',
                                                             tokenizer='NLTK',
                                                             output_style='FULL',
                                                             binary_label=True)
            self.training_data = self.biocreative_dataset.dataset[0]
            self.training_labels = self.biocreative_dataset.dataset[1]
            self.training_size = len(self.training_data)

    def parse_dataset(self):

        if self.dataset_name == 'BioCreative':
            for pair in self.training_data:

