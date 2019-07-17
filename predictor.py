import datetime
import configparser


class Predictor(object):
    """ Predictor class is used to conduct experiments with selected models to selected dataset.

    The __init__ method initialize an object of a Predictor class.

    """
    def __init__(self, caller, model, predictor_id, data_interface):
        self.caller = caller  # cv_run, grid_search or single.
        self.model = model  # bilstm, kimcnn.
        self.id = predictor_id  # id assigned to predictor by caller.
        self.data_interface = data_interface  # data interface object

        # 1. create a report file to output results.
        report_name = self.caller + "_" + self.model + "_" + self.id
        report_file = open(report_name, "w+")
        line = "Time: {}".format(str(datetime.datatime.now()))
        print(line)
        report_file.write(line+'\n')
        line = "Caller: {}".format(self.caller)
        print(line)
        report_file.write(line+'\n')
        line = "Model: {}".format(self.model)
        print(line)
        report_file.write(line+'\n')

        # 2. obtain parameters
        config = configparser.ConfigParser()
        config.read('config.ini')

        # 2.1. read model hyper parameters from config.ini file
        # 2.2. read common parameters
        section_name = 'BASE'

        # 2.2.1 batch_size
        batch_size = int(config[section_name]['batch_size'])
        line = "Batch Size: {}".format(str(batch_size))
        print(line)
        report_file.write(line+"\n")

        # 2.2.2. epoch_size
        num_epoch = int(config[section_name]['num_epoch'])
        line = "Number of Epoch: {}".format(str(num_epoch))
        print(line)
        report_file.write(line + "\n")

        # 2.2.3. learning rate
        learning_rate = int(config[section_name]['learning_rate'])
        line = "Learning Rate: {}".format(str(learning_rate))
        print(line)
        report_file.write(line+'\n')

        # 2.2.4. error function
        error_function = config[section_name]['error_function']
        if error_function == 'unweighted_ce':
            line = "Error Function: {}".format('Unweighted Cross Entropy')
            print(line)
            report_file.write(line+'\n')
        elif error_function == 'weighted':
            line = 'Error Function: {}'.format('Weighted Cross Entropy')
            print(line)
            report_file.write(line+'\n')
            # TODO: add weights to the config.ini file and read them.
            # TODO: weighted cross entropy is not ready to use now.

        # 2.3. read embeddings parameters
        section_name = 'EMBEDDINGS'

        # 2.3.1 word embeddings
        word_embedding_type = config[section_name]['word_embedding_type']
        line = 'Word Embedding Type: {}'.format(word_embedding_type)
        print(line)
        report_file.write(line+'\n')

        word_embedding_size = config[section_name]['word_embedding_size']
        line = 'Word Embedding Size: {}'.format(word_embedding_size)
        print(line)
        report_file.write(line+'\n')
        # TODO: only 200 is valid word embedding size currently.

        # 2.3.2 position embeddings
        position_embedding_flag = config[section_name]['position_embedding_flag']
        line = 'Position Embedding Flag: {}'.format(position_embedding_flag)
        print(line)
        report_file.write(line+'\n')
        position_embedding_flag = True if position_embedding_flag == 'true' else False
        position_embedding_size = 0
        if position_embedding_flag:
            position_embedding_size = int(config[section_name]['position_embedding_size'])
            line = 'Position Embedding Size: {}'.format(str(position_embedding_size))
            print(line)
            report_file.write(line+'\n')

        # 2.3.3 pos tag embeddings
        pos_tag_flag = config[section_name]['pos_tag_embedding_flag']
        line = 'POS Tag Embedding Flag: {}'.format(pos_tag_flag)
        print(line)
        report_file.write(line+'\n')
        pos_tag_flag = True if pos_tag_flag == 'true' else False
        pos_tag_size = 0
        if pos_tag_flag:
            pos_tag_size = int(config[section_name]['pos_tag_embedding_size'])
            line = 'POS Tag Embedding Size: {}'.format(str(pos_tag_size))
            print(line)
            report_file.write(line+'\n')

        # 2.3.4 iob tag embeddings
        iob_tag_flag = config[section_name]['iob_embedding_flag']
        line = 'IOB Tag Embedding Flag: {}'.format(iob_tag_flag)
        print(line)
        report_file.write(line+'\n')
        iob_tag_flag = True if iob_tag_flag == 'true' else False
        iob_tag_size = 0
        if iob_tag_flag:
            iob_tag_size = int(config[section_name]['iob_embedding_size'])
            line = 'IOB Embedding Size: {}'.format(str(iob_tag_size))
            print(line)
            report_file.write(line+'\n')

        # 3. data interface
        line = '\nData Interface'
        print(line)
        report_file.write(line+'\n')

        training_dict = self.data_interface['training']
        training_set_size = len(training_dict['data'])
        line = 'Training Set Size: {}'.format(str(training_set_size))
        print(line)
        report_file.write(line+'\n')
        max_train_seq_length = training_dict['max_seq_length']
        line = 'Maximum Training Sequence Length: {}'.format(str(max_train_seq_length))
        print(line)
        report_file.write(line+'\n')

        development_dict = self.data_interface['development']
        development_set_size = len(development_dict['data'])
        line = 'Development Set Size: {}'.format(str(development_set_size))
        print(line)
        report_file.write(line+'\n')
        max_dev_seq_length = development_dict['max_seq_length']
        line = 'Maximum Development Sequence Length: {}'.format(str(max_dev_seq_length))
        print(line)
        report_file.write(line+'\n')

        test_dict = self.data_interface['test']
        test_set_size = len(test_dict['data'])
        line = 'Test Set Size: {}'.format(str(test_set_size))
        print(line)
        report_file.write(line+'\n')
        max_test_seq_length = test_dict['max_seq_length']
        line = 'Maximum Test Sequence Length: {}'.format(str(max_test_seq_length))
        print(line)
        report_file.write(line+'\n')

        embedding_matrix = self.data_interface.embeddings
        embedding_dimentsion = embedding_matrix.shape[1]
