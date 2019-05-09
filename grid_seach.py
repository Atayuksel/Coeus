import itertools

range_batch_size = [50, 100, 200]
range_num_epoch = [20, 50]
range_hidden_unit = [1024, 2048, 4096]
range_learning_rate = [0.01, 0.001]
range_embedding_size = [100, 200, 300]

range_conv_filter_size_height = [2, 3, 4, 5, 10, 20, 50]
range_conv_filter_size_width = [2, 3, 4, 5, 10, 20, 50]

range_conv_filter_out_1 = [16, 32, 64, 128]
range_conv_filter_out_2 = [16, 32, 64, 128]
range_conv_filter_out_3 = [16, 32, 64, 128]

range_conv_filter_stride_height = [1]
range_conv_filter_stride_width = [1]

range_pooling_filter_size_height = [2, 3, 4, 5]
range_pooling_filter_size_width = [2, 3, 4, 5]

parameter_list = [range_batch_size, range_num_epoch, range_hidden_unit, range_learning_rate, range_embedding_size,
                  range_conv_filter_size_height, range_conv_filter_size_width,
                  range_conv_filter_out_1, range_conv_filter_out_2, range_conv_filter_out_3,
                  range_conv_filter_stride_height, range_conv_filter_stride_width,
                  range_pooling_filter_size_height, range_pooling_filter_size_width]

parameter_combinations = list(itertools.product(*parameter_list))
num_parameters = len(parameter_combinations)
print('asdaskldj')