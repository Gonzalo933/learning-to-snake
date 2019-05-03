import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import numpy as np


class NeuralNetwork:
    def __init__(self, D_problem, n_classes, dim_hidden_layers=[3], learning_rate=1e-3):
        self.input_layer = tf.placeholder(
            shape=[None, D_problem], dtype=tf.float32, name="X"
        )
        last_layer = self.input_layer
        for dim_layer in dim_hidden_layers:
            hidden_layer = tf_layers.fully_connected(
                last_layer, dim_layer, activation_fn=tf.nn.relu, biases_initializer=None
            )
            last_layer = hidden_layer
        self.logits = tf_layers.fully_connected(
            last_layer, n_classes, activation_fn=None, biases_initializer=None
        )
        # As softmax_cross_entropy_with_logits_v2 requires the logits
        #  before passing them to the softmax activation func.
        self.output_layer = tf.nn.softmax(self.logits)

        self.actual_actions = tf.placeholder(
            shape=[None, 4], dtype=tf.int32, name="actions"
        )
        self.game_rewards = tf.placeholder(shape=[None], dtype=tf.float32, name="rewards")

        # self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        #    labels=self.actual_actions, logits=self.output_layer_before_softmax
        # )
        # self.policy_loss = -tf.reduce_mean(self.loss * self.game_rewards)
        self.policy_loss = -1 * tf.losses.softmax_cross_entropy(
            onehot_labels=self.actual_actions,
            logits=self.logits,
            weights=self.game_rewards,
            reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE,
        )
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
