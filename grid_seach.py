import itertools
import configparser
import os
import time

range_batch_size = [50]
range_num_epoch = [30]
range_hidden_unit = [2048]
range_learning_rate = [0.001]
range_embedding_size = [100, 300]

range_conv_filter_size_height = [2, 4]
range_conv_filter_size_width = [10, 50]

range_conv_filter_out_1 = [64, 128]
range_conv_filter_out_2 = [64, 128]
range_conv_filter_out_3 = [64, 128]

range_conv_filter_stride_height = [1]
range_conv_filter_stride_width = [1]

range_pooling_filter_size_height = [2, 4]
range_pooling_filter_size_width = [2, 4]

parameter_list = [range_batch_size, range_num_epoch, range_hidden_unit, range_learning_rate, range_embedding_size,
                  range_conv_filter_size_height, range_conv_filter_size_width,
                  range_conv_filter_out_1, range_conv_filter_out_2, range_conv_filter_out_3,
                  range_conv_filter_stride_height, range_conv_filter_stride_width,
                  range_pooling_filter_size_height, range_pooling_filter_size_width]

parameter_combinations = list(itertools.product(*parameter_list))
num_parameters = len(parameter_combinations)
filtered_parameter_combinations = []
for combination in parameter_combinations:
    conv_filter_out_1 = combination[7]
    conv_filter_out_2 = combination[8]
    conv_filter_out_3 = combination[9]
    if conv_filter_out_1 <= conv_filter_out_2 <= conv_filter_out_3:
        filtered_parameter_combinations.append(combination)
filtered_num_parameters = len(filtered_parameter_combinations)

config_parameter = configparser.ConfigParser()
config_parameter.read('config.ini')
config_parameter.set("HYPERPARAMETERS", "RUN_TYPE", "GRIDSEARCH")
config_parameter.set('HYPERPARAMETERS', 'REPORT_FILE_NAME', 'gridsearch_report.txt')
config_parameter.set('HYPERPARAMETERS', 'MODEL', 'CNN')

for parameters in filtered_parameter_combinations:
    # write parameters to config.ini file for driver
    config_parameter.set('HYPERPARAMETERS', 'BATCH_SIZE', str(parameters[0]))
    config_parameter.set('HYPERPARAMETERS', 'NUM_EPOCH', str(parameters[1]))
    config_parameter.set('HYPERPARAMETERS', 'NUM_HIDDEN_UNIT', str(parameters[2]))
    config_parameter.set('HYPERPARAMETERS', 'LEARNING_RATE', str(parameters[3]))
    config_parameter.set('HYPERPARAMETERS', 'EMBEDDING_SIZE', str(parameters[4]))
    config_parameter.set('HYPERPARAMETERS', 'CONV_FILTER_SIZE_HEIGHT', str(parameters[5]))
    config_parameter.set('HYPERPARAMETERS', 'CONV_FILTER_SIZE_WIDTH', str(parameters[6]))
    config_parameter.set('HYPERPARAMETERS', 'CONV_FILTER_OUT_1', str(parameters[7]))
    config_parameter.set('HYPERPARAMETERS', 'CONV_FILTER_OUT_2', str(parameters[8]))
    config_parameter.set('HYPERPARAMETERS', 'CONV_FILTER_OUT_3', str(parameters[9]))
    config_parameter.set('HYPERPARAMETERS', 'CONV_FILTER_STRIDE_HEIGHT', str(parameters[10]))
    config_parameter.set('HYPERPARAMETERS', 'CONV_FILTER_STRIDE_WIDTH', str(parameters[11]))
    config_parameter.set('HYPERPARAMETERS', 'POOLING_FILTER_SIZE_HEIGHT', str(parameters[12]))
    config_parameter.set('HYPERPARAMETERS', 'POOLING_FILTER_SIZE_WIDTH', str(parameters[13]))

    # Writing our configuration file to 'example.ini'
    with open("config.ini", 'w') as configfile:
        config_parameter.write(configfile)

    cmd = os.path.join(os.getcwd(), "driver.py")
    os.system('{} {}'.format('python3', cmd))

    time.sleep(3)