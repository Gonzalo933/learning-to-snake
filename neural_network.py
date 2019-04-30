import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import numpy as np


def my_cross_entropy(y_true, y_pred):
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


class NeuralNetwork:
    def __init__(self, D_problem, n_classes, dim_hidden_layers=[3], learning_rate=1e-4):
        self.input_layer = tf.placeholder(shape=[None, D_problem], dtype=tf.float32)
        last_layer = self.input_layer
        for dim_layer in dim_hidden_layers:
            hidden_layer = tf_layers.fully_connected(
                last_layer, dim_layer, activation_fn=tf.nn.relu, biases_initializer=None
            )
            last_layer = hidden_layer
        self.output_layer_before_softmax = tf_layers.fully_connected(
            last_layer, n_classes, activation_fn=tf.nn.softmax, biases_initializer=None
        )
        # As softmax_cross_entropy_with_logits_v2 requires the logits
        #  before passing them to the softmax activation func.
        self.output_layer = tf.nn.softmax(self.output_layer_before_softmax)

        self.actual_actions = tf.placeholder(shape=[None, 4], dtype=tf.int32)
        self.game_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.actual_actions, logits=self.output_layer_before_softmax
        )
        self.policy_loss = -tf.reduce_mean(self.loss * self.game_rewards)

        w_variables = tf.trainable_variables()
        self.gradients = []
        for indx, w in enumerate(w_variables):
            w_holder_var = tf.placeholder(tf.float32, name="w_" + str(indx))
            self.gradients.append(w_holder_var)

        self.all_gradients = tf.gradients(self.policy_loss, w_variables)
        # optimizer = tf.train.RMSPropOptimizer(decay = decay_rate, learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.apply_grads = optimizer.apply_gradients(zip(self.gradients, w_variables))

        self.saver = tf.train.Saver()

