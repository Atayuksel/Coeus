import dataset
import random
import math
import pickle
import os.path
import configparser
import numpy as np
import data_interface as di
import predictor
from scipy.special import softmax


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


def calculate_map(model_prediction_logit, model_true_values, k):
    """
    calculate the mean average precision of the model wrt model logits and model true values
    :param model_prediction_logit: model prediction logits for each run
    :param model_true_values: model true values for each run
    :return: mean average precision
    """

    model_average_precisions = []

    # for each run
    for i in range(len(model_prediction_logit)):
        # obtain logit values and true values for a run
        run_logit = model_prediction_logit[i]
        run_truth = model_true_values[i]

        # convert truth values to one-hot matrix
        run_truth = convert_one_hot(run_truth)

        # apply softmax to logits in axis 1.
        run_logit = softmax(run_logit, axis=1)

        # concatenate logit to true values for sorting and keeping aligned.
        run_logit_true = np.concatenate((run_logit, run_truth), axis=1)

        # sort the concatenated matrix wrt second column which is probability of predicting 1.
        run_logit_true = run_logit_true[np.array((-run_logit_true[:, 1]).argsort(axis=0).tolist()).ravel()]

        # split concatenated matrix to obtain logits and true values separately.
        run_logit = run_logit_true[:, :2]
        run_truth = run_logit_true[:, 2:]

        # create correct predictions array
        run_correct_predictions = []
        if k == 0:
            for j in range(len(run_logit)):
                single_logit = run_logit[j]
                single_true_value = run_truth[j]
                if (single_logit[1] > single_logit[0]) == (single_true_value[1] > single_true_value[0]):
                    run_correct_predictions.append(1)
                else:
                    run_correct_predictions.append(0)

        # calculate average precision for top k
        # if k is 0, calculate average precision for all predictions.
        run_average_precisions = []
        for j in range(len(run_correct_predictions)):
            prediction = run_correct_predictions[j]
            if prediction == 1:
                run_average_precisions.append((len(run_average_precisions)+1)/(j+1))
        run_average_precision = np.sum(run_average_precisions)
        run_average_precision = run_average_precision / len(run_average_precisions)

        model_average_precisions.append(run_average_precision)

    model_average_precision = np.sum(model_average_precisions)
    model_average_precision = model_average_precision / len(model_average_precisions)
    return model_average_precision


def cv_t_test(scores):
    """
    calculate t value wrt to scores matrix
    :param scores: contains precision values of the models for each iteration.
    :return: t value
    """

    iteration_number = int(scores.shape[0]/2)  # iteration value

    # calculate the differences of precision between two models.
    all_difference = scores[:, 0] - scores[:, 1]

    # mean of differences in each iteration
    all_diff_mean = np.zeros([iteration_number])
    all_variance = np.zeros([iteration_number])

    # for each iteration
    for iteration in range(iteration_number):
        # obtain first fold values
        fold1_diff = all_difference[iteration*2+0]
        # obtain second fold values
        fold2_diff = all_difference[iteration*2+1]

        # calculate mean of difference for each fold in the iteration
        iteration_diff_mean = (fold1_diff + fold2_diff) / 2
        # store mean of difference for the iteration
        all_diff_mean[iteration] = iteration_diff_mean

        # calculate variance for the iteration
        variance = (fold1_diff - iteration_diff_mean)**2 + (fold2_diff - iteration_diff_mean)**2
        # store the variance for the iteration
        all_variance[iteration] = variance

    # calculate t value
    t = np.sum(all_variance) * (1/iteration_number)
    t = math.sqrt(t)
    t = all_difference[0] / t
    return t


def best_classifier(scores):
    """
    selects the best model wrt f1-measure scores.
    :param scores: f1-measure scores of the models for each iteration
    :return: 0 for the first model and 1 for the second model.
    """

    sum_scores = np.sum(scores, axis=0)
    repetation_number = sum_scores.shape[0]
    sum_scores = sum_scores/repetation_number
    return sum_scores
    # if sum_scores[0] > sum_scores[1]:
    #     return 0
    # else:
    #     return 1


def convert_one_hot(data):
    result_data = []
    for label in data:
        np_label = np.zeros(2)
        np_label[label] = 1
        result_data.append(np_label)
    return result_data


repetition_number = 2
fold_number = 2

# create a report file
cv_report_file = open('cv_report_file', 'w')

# initialize config parser
config = configparser.ConfigParser()
config.read('config.ini')

# read dataset parameters
root_directory = config['INTERFACE']['root_directory']
word_tokenizer = config['INTERFACE']['word_tokenizer']
relation_type = config['INTERFACE']['relation_type']
interface_root_dir = config['INTERFACE']['root_directory']

# set directory for prepared dataset in interface root directory
training_data_dir = interface_root_dir + '/cv_training_data.pkl'
dev_data_dir = interface_root_dir + '/cv_development_data.pkl'
test_data_dir = interface_root_dir + '/cv_test_data.pkl'

# create dataset
bc_dataset = dataset.BioCreative(root_directory=root_directory,
                                 tokenizer=word_tokenizer,
                                 relation_type=relation_type)

# parse training and development dataset
raw_dataset = bc_dataset.dataset
instances = raw_dataset[0] + raw_dataset[2] + raw_dataset[4]
labels = raw_dataset[1] + raw_dataset[3] + raw_dataset[5]
data = list(zip(instances, labels))

# parse test dataset
# test_instances = raw_dataset[4]
# test_labels = raw_dataset[5]
# test_data = list(zip(test_instances, test_labels))

# create cv matrix
precision_matrix = np.zeros([repetition_number*fold_number, 2])
recall_matrix = np.zeros([repetition_number*fold_number, 2])
f1_measure_matrix = np.zeros([repetition_number*fold_number, 2])

a_predictions = []
a_truth_values = []
a_logits = []

b_predictions = []
b_truth_values = []
b_logits = []

for i in range(repetition_number):

    # shuffle data
    data = random.sample(data, len(data))

    # split data into n-fold
    data_chunks = split_list(data, fold_number)

    for j in range(fold_number):

        testSetIdx = j
        testSet = data_chunks[j]
        trainSet = obtain_train_sets(data_chunks, j)

        if os.path.exists(training_data_dir):
            os.remove(training_data_dir)
        # if os.path.exists(dev_data_dir):
        #     os.remove(dev_data_dir)
        if os.path.exists(test_data_dir):
            os.remove(test_data_dir)

        with open(training_data_dir, 'wb') as f:
            pickle.dump(trainSet, f)
        # with open(dev_data_dir, 'wb') as f:
        #     pickle.dump(devSet, f)
        with open(test_data_dir, 'wb') as f:
            pickle.dump(testSet, f)

        # update configuration file
        config.set("MODEL", "train_word_embeddings", "true")

        # write to configuration file
        with open("config.ini", 'w') as configfile:
            config.write(configfile)

        # create data interface
        data_interface = di.DataInterface()

        # create predictor model
        predictor_id = str(i)+str(j)+str(1)

        # create predictor directory
        current_path = os.getcwd()
        predictor_report_path = current_path + '/' + 'cv_reports' + '/' + predictor_id
        if not os.path.exists(predictor_report_path):
            os.makedirs(predictor_report_path)

        # write to the report file
        cv_report_file.write("\nPREDICTOR_ID: {}\n".format(predictor_id))
        data_interface.write_information(cv_report_file)

        model_predictor = predictor.Predictor(predictor_id=predictor_id,
                                              data_interface=data_interface,
                                              report_directory=predictor_report_path,
                                              development_set_flag=False,
                                              test_set_flag=True,
                                              early_stopping_set='test')

        # train model
        model_predictor.train(min_epoch_number=30)

        # obtain results
        dev_results = model_predictor.get_test_results()
        dev_metric = dev_results[0]
        dev_prediction = dev_results[1]
        dev_logits = dev_results[2]
        dev_truth_value = dev_results[3]

        # update predictions and results
        a_predictions.append(dev_prediction)
        a_truth_values.append(dev_truth_value)
        a_logits.append(dev_logits)

        # update cv matrix
        precision_matrix[i*fold_number+j, 0] = dev_metric[0]
        recall_matrix[i*fold_number+j, 0] = dev_metric[1]
        f1_measure_matrix[i*fold_number+j, 0] = dev_metric[2]

        cv_report_file.write("Best: Precision: {}, Recall: {}, F1-Measure: {}\n".format(dev_metric[0],
                                                                                        dev_metric[1],
                                                                                        dev_metric[2]))

        del model_predictor
        del data_interface

        # 2nd Model
        config.set("MODEL", "train_word_embeddings", "false")
        with open("config.ini", 'w') as configfile:
            config.write(configfile)

        data_interface = di.DataInterface()

        predictor_id = str(i) + str(j) + str(2)

        current_path = os.getcwd()
        predictor_report_path = current_path + '/' + 'cv_reports' + '/' + predictor_id
        if not os.path.exists(predictor_report_path):
            os.makedirs(predictor_report_path)

        # write to the report file
        cv_report_file.write("\nPREDICTOR_ID: {}\n".format(predictor_id))
        data_interface.write_information(cv_report_file)

        model_predictor = predictor.Predictor(predictor_id=predictor_id,
                                              data_interface=data_interface,
                                              report_directory=predictor_report_path,
                                              development_set_flag=False,
                                              test_set_flag=True,
                                              early_stopping_set='test')

        # train model
        model_predictor.train(min_epoch_number=30)

        # obtain results
        dev_results = model_predictor.get_test_results()
        dev_metric = dev_results[0]
        dev_prediction = dev_results[1]
        dev_logits = dev_results[2]
        dev_truth_value = dev_results[3]

        # update predictions and results
        b_predictions.append(dev_prediction)
        b_truth_values.append(dev_truth_value)
        b_logits.append(dev_logits)

        # update cv matrix
        precision_matrix[i * fold_number + j, 1] = dev_metric[0]
        recall_matrix[i * fold_number + j, 1] = dev_metric[1]
        f1_measure_matrix[i * fold_number + j, 1] = dev_metric[2]

        cv_report_file.write("Best: Precision: {}, Recall: {}, F1-Measure: {}\n".format(dev_metric[0],
                                                                                        dev_metric[1],
                                                                                        dev_metric[2]))

        del model_predictor
        del data_interface


cv_report_file.write("\n 5x2 Cross Validation Results:\n")
average_f1_measure = best_classifier(f1_measure_matrix)

a_mean_average_value = calculate_map(a_logits, a_truth_values, 0)
cv_report_file.write("\n First Model mAP: {}\n".format(a_mean_average_value))
cv_report_file.write("First Model Average F1-measure: {}\n".format(average_f1_measure[0]))

b_mean_average_value = calculate_map(b_logits,  b_truth_values, 0)
cv_report_file.write("\n Second Model mAP: {}\n".format(b_mean_average_value))
cv_report_file.write("Second Model Average F1-measure: {}\n".format(average_f1_measure[1]))

cv_report_file.write("\nPaired T Test:\n")
t_value = cv_t_test(f1_measure_matrix)
cv_report_file.write("T value: {}".format(t_value))
