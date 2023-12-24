import tensorflow as tf


class PolynomialNeuralLayer(tf.keras.layers.Layer):
    def __init__(self, output_dimension, polynomial_order, initial_weights=None, **kwargs):
        super(PolynomialNeuralLayer, self).__init__(**kwargs)
        self.polynomial_order = polynomial_order
        self.output_dimension = output_dimension
        self.initial_weights = initial_weights or []
        self.polynomial_weights = []
        self.polynomial_dimensions = []

    def build(self, input_shape):
        input_dimension = input_shape[1]
        self.polynomial_dimensions = [input_dimension ** (n + 1) for n in range(self.polynomial_order)]

        initial_values = []
        if self.initial_weights:
            transposed_weights = [weights.T for weights in self.initial_weights]
            bias_initial_value = transposed_weights[0]
            initial_values = transposed_weights[1:]
        else:
            bias_initial_value = tf.zeros_initializer()(shape=(1, self.output_dimension))
            initial_values.append(tf.eye(self.polynomial_dimensions[0], self.output_dimension))
            for i in range(1, self.polynomial_order):
                shape = (self.polynomial_dimensions[i], self.output_dimension)
                initial_values.append(tf.zeros_initializer()(shape=shape))

        if self.polynomial_order + 1 > len(self.initial_weights):
            shape = (self.polynomial_dimensions[self.polynomial_order - 1], self.output_dimension)
            initial_values.append(tf.zeros_initializer()(shape=shape))

        self.bias = tf.Variable(initial_value=bias_initial_value, dtype=tf.float32)

        for i, value in enumerate(initial_values):
            var = tf.Variable(initial_value=value, dtype=tf.float32, name=f'W_{i + 1}')
            self.polynomial_weights.append(var)

        self._trainable_weights.extend([kernel for kernel in self.polynomial_weights if kernel.trainable])

    def call(self, inputs):
        result = self.bias
        input_degree = tf.ones_like(inputs[:, 0:1])

        for i in range(self.polynomial_order):
            input_degree = tf.einsum('bi,bj->bij', inputs, input_degree)
            input_degree = tf.reshape(input_degree, [-1, self.polynomial_dimensions[i]])
            result = result + tf.matmul(input_degree, self.polynomial_weights[i])

        return result

    def get_config(self):
        config = super().get_config()
        config.update({
            "polynomial_order": self.polynomial_order,
            "output_dimension": self.output_dimension,
            "initial_weights": self.initial_weights,
        })
        return config
