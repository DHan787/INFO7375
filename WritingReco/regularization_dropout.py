import tensorflow as tf

class Regularization:
    def __init__(self, lambda_val):
        self.lambda_val = lambda_val

    def apply_regularization(self, model):
        regularization_loss = 0
        for layer in model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                regularization_loss += self.lambda_val * tf.reduce_sum(layer.kernel_regularizer(layer.kernel))
        return regularization_loss


class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def apply_dropout(self, inputs, training):
        if training:
            return tf.nn.dropout(inputs, rate=self.dropout_rate)
        else:
            return inputs
