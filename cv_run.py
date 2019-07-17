import bc_dataset
import random
import math
import pickle
import os.path
import configparser
import time

def split_list(data, fold):
    size = len(data)
    chunks = []
    chunk_size = math.floor(size / fold)
    for i in range(fold-1):
        tmp = data[:chunk_size]
        chunks.append(tmp)
    tmp = data[(chunk_size*(fold_number-1)):]
    chunks.append(tmp)
    return chunks


def obtain_train_sets(data, test_idx):
    chunk_number = len(data)
    training_data = []
    for i in range(chunk_number):
        if i != test_idx:
            training_data = training_data + data[i]
    return training_data


repetition_number = 5
fold_number = 2

config = configparser.ConfigParser()
config.read('config.ini')
training_data_dir = config['CV']['cv_training_data_dir']
test_data_dir = config['CV']['cv_test_data_dir']

bc_dataset = bc_dataset.BioCreativeData(input_root='dataset',
                                        output_root='output/bc_dataset',
                                        sent_tokenizer='NLTK',
                                        binary_label=True)

raw_dataset = bc_dataset.dataset
instances = raw_dataset[0] + raw_dataset[2]
labels = raw_dataset[1] + raw_dataset[3]
data = list(zip(instances, labels))

config_parameter = configparser.ConfigParser()
config_parameter.read('config.ini')
config_parameter.set("BASE", "run_type", "cv_run")
config_parameter.set('BASE', 'development_set', 'true')
config_parameter.set('BASE', 'test_set', 'false')


for i in range(repetition_number):
    data = random.sample(data, len(data))
    data_chunks = split_list(data, fold_number)
    for j in range(fold_number):
        testSetIdx = j
        testSet = data_chunks[j]
        trainSet = obtain_train_sets(data_chunks, j)

        if os.path.exists(training_data_dir):
            os.remove(training_data_dir)
        if os.path.exists(test_data_dir):
            os.remove(test_data_dir)

        with open(training_data_dir, 'wb') as f:
            pickle.dump(trainSet, f)
        with open(test_data_dir, 'wb') as f:
            pickle.dump(testSet, f)

        # 1- update the config.ini file
        config_parameter.set('BILSTM', 'lstm_hidden_unit', '128')

        with open("config.ini", 'w') as configfile:
            config_parameter.write(configfile)

        # 2- run 1st model with training set and test with development set
        cmd = os.path.join(os.getcwd(), "driver.py")
        os.system('{} {}'.format('python3', cmd))
        time.sleep(3)

        # 3- obtain the results


        # 4- calculate the results

        # 5- update the config.ini file
        # 6- run 2nd model with training set and test with test set
        # 7- obtain the results
        # 8- calculate the results

# 9- report the paired 5x2 cv t test results