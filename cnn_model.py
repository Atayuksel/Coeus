import tensorflow as tf
import functools

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class CNNModel:
    def __init__(self, data, target):
        self.data = data
        self.target = target

        # network variables (weights, biases, embeddings)
        self.weights = {
            'out': tf.Variable(tf.random_normal([2 * self.num_hidden, target_size.value]))}

    def conv2d(self, data, ):