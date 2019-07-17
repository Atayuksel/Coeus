import itertools
import configparser
import os
import time
import pickle
import numpy as np


def compare_models(true_values, classifier0, classifier1):
    contingencyTable = np.zeros((2, 2))
    classifier0Table = np.zeros((2, 2))
    classifier1Table = np.zeros((2, 2))

    for i in range(len(true_values)):
        true_value = true_values[i]
        classifier0_value = classifier0[i]
        classifier1_value = classifier1[i]
        if true_value == classifier0_value:
            if true_value == classifier1_value:
                contingencyTable[0][0] = contingencyTable[0][0] + 1
            else:
                contingencyTable[0][1] = contingencyTable[0][1] + 1
        else:
            if true_value == classifier1_value:
                contingencyTable[1][0] = contingencyTable[1][0] + 1
            else:
                contingencyTable[1][1] = contingencyTable[1][1] + 1

        if true_value == 0:
            if classifier0_value == 0:
                classifier0Table[1][1] = classifier0Table[1][1] + 1
            else:
                classifier0Table[1][0] = classifier0Table[1][0] + 1

            if classifier1_value == 0:
                classifier1Table[1][1] = classifier1Table[1][1] + 1
            else:
                classifier1Table[1][0] = classifier1Table[1][0] + 1
        else:
            if classifier0_value == 0:
                classifier0Table[0][1] = classifier0Table[0][1] + 1
            else:
                classifier0Table[0][0] = classifier0Table[0][0] + 1

            if classifier1_value == 0:
                classifier1Table[0][1] = classifier1Table[0][1] + 1
            else:
                classifier1Table[0][0] = classifier1Table[0][0] + 1

    classifier0_precision = classifier0Table[0][0] / (classifier0Table[0][0] + classifier0Table[1][0])
    classifier0_recall = classifier0Table[0][0] / (classifier0Table[0][0] + classifier0Table[0][1])
    classifier0_measure = 2 * ((classifier0_precision * classifier0_recall) / (classifier0_precision + classifier0_recall))

    classifier1_precision = classifier1Table[0][0] / (classifier1Table[0][0] + classifier1Table[1][0])
    classifier1_recall = classifier1Table[0][0] / (classifier1Table[0][0] + classifier1Table[0][1])
    classifier1_measure = 2 * ((classifier1_precision * classifier1_recall) / (classifier1_precision + classifier1_recall))

    statistic = (contingencyTable[0][1] - contingencyTable[1][0]) ** 2 / (contingencyTable[0][1] + contingencyTable[0][1])
    alpha = 0.05
    if statistic <= alpha:
        if classifier1_measure > classifier0_measure:
            return 1
        else:
            return 0
    else:
        return 0


# read grid search space
gss_file = open('grid_search_values.txt', 'r')
gss_lines = gss_file.readlines()

gs_map = []
for line in gss_lines:
    gs_parameter = line.split()
    label = gs_parameter[:2]
    space = gs_parameter[2:]
    gs_map.append(list((label, space)))

parameter_list = []
for item in gs_map:
    label = item[0]
    space = item[1]
    parameter_list.append(space)

parameter_combinations = list(itertools.product(*parameter_list))
num_parameters = len(parameter_combinations)

config_parameter = configparser.ConfigParser()
config_parameter.read('config.ini')
config_parameter.set("BASE", "run_type", "grid_search")
config_parameter.set('BASE', 'model_type', 'bilstm')
config_parameter.set('BASE', 'development_set', 'true')
config_parameter.set('BASE', 'test_set', 'false')
config_parameter.set('BASE', 'gridsearch_report_file_name', 'test_case_grid_search_results_1.txt')

startFlag = True
bestParameters = []
truthLabels = []
bestPredictedValues = []

for parameters in parameter_combinations:

    # write parameters to config.ini file for driver
    for i in range(len(gs_map)):
        config_parameter.set(gs_map[i][0][0], gs_map[i][0][1], str(parameters[i]))

    # Writing our configuration file to 'example.ini'
    with open("config.ini", 'w') as configfile:
        config_parameter.write(configfile)

    cmd = os.path.join(os.getcwd(), "driver.py")
    os.system('{} {}'.format('python3', cmd))

    time.sleep(3)

    if startFlag:
        bestParameters = parameters
        startFlag = False
        truthLabels = pickle.load(open("test_set_truth_values.pkl", "rb"))
        bestPredictedValues = pickle.load(open("test_set_predicted_values.pkl", "rb"))
    else:
        currentPredictedValues = pickle.load(open("test_set_predicted_values.pkl", "rb"))
        compare_flag = compare_models(truthLabels, bestPredictedValues, currentPredictedValues)
        if compare_flag == 1:
            bestPredictedValues = currentPredictedValues
            bestParameters = parameters

    print("Best Model Parameters:")
    print("{}: {}".format(gs_map[i][0][1], str(parameters[i])))