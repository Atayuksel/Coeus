import configparser
import data_interface as di
import os
import predictor
import numpy as np
from scipy.special import softmax


def convert_one_hot(data):
    result_data = []
    for label in data:
        np_label = np.zeros(2)
        np_label[label] = 1
        result_data.append(np_label)
    return result_data


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


def calculate_ap(model_prediction_logit, model_true_values, k):

    # convert truth values to one-hot matrix
    run_truth = convert_one_hot(model_true_values)

    # apply softmax to logits in axis 1.
    run_logit = softmax(model_prediction_logit, axis=1)

    # concatenate logit to true values for sorting and keeping aligned.
    run_logit_true = np.concatenate((run_logit, run_truth), axis=1)
    # sort the concatenated matrix wrt second column which is probability of predicting 1.
    run_logit_true = run_logit_true[np.array((-run_logit_true[:, 1]).argsort(axis=0).tolist()).ravel()]
    # split concatenated matrix to obtain logits and true values separately.
    run_logit = run_logit_true[:, :2]
    run_truth = run_logit_true[:, 2:]

    # create correct predictions array
    # if k is 0, calculate average precision for all predictions.
    run_correct_predictions = []
    for i in range(len(run_logit)):
        single_logit = run_logit[i]
        single_true_value = run_truth[i]
        if single_logit[1] > k:
            if (single_logit[1] > single_logit[0]) == (single_true_value[1] > single_true_value[0]):
                run_correct_predictions.append(1)
            else:
                run_correct_predictions.append(0)

    run_average_precisions = []
    for j in range(len(run_correct_predictions)):
        prediction = run_correct_predictions[j]
        if prediction == 1:
            run_average_precisions.append((len(run_average_precisions) + 1) / (j + 1))
    run_average_precision = np.sum(run_average_precisions)
    run_average_precision = run_average_precision / len(run_average_precisions)

    return run_average_precision


# initialize config parser
config = configparser.ConfigParser()
config.read('config.ini')

# set root directory
config.set("INTERFACE", "root_directory", "/content/drive/My Drive/Coeus/colab/dataset")
with open("config.ini", 'w') as configfile:
    config.write(configfile)

# read dataset parameters
interface_root_dir = config['INTERFACE']['root_directory']

# create data interface
data_interface = di.DataInterface()

# create predictor
predictor_id = '1'

current_path = os.getcwd()
predictor_report_path = current_path + '/' + 'single_reports' + '/' + predictor_id
if not os.path.exists(predictor_report_path):
    os.makedirs(predictor_report_path)

model_predictor = predictor.Predictor(predictor_id=predictor_id,
                                      data_interface=data_interface,
                                      report_directory=predictor_report_path,
                                      development_set_flag=False,
                                      test_set_flag=True,
                                      early_stopping_set='test')

model_predictor.train(min_epoch_number=1)

# obtain results
test_results = model_predictor.get_test_results()
test_metric = test_results[0]
test_prediction = test_results[1]
test_logits = test_results[2]
test_truth_value = test_results[3]

k_values = [0.9, 0.8, 0.7, 0.6, 0.5]
for k in k_values:
    k_ap = calculate_ap(test_logits, test_truth_value, k)
    print('Average Precision for {}: {}'.format(str(k), str(k_ap)))