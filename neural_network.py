import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import numpy as np


def my_cross_entropy(y_true, y_pred):
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


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
        self.output_layer_before_softmax = tf_layers.fully_connected(
            last_layer, n_classes, activation_fn=tf.nn.softmax, biases_initializer=None
        )
        # As softmax_cross_entropy_with_logits_v2 requires the logits
        #  before passing them to the softmax activation func.
        self.output_layer = tf.nn.softmax(self.output_layer_before_softmax)

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
            logits=self.output_layer_before_softmax,
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


class ConvolutionalNeuralNetwork:
    def __init__(self, D_problem, n_classes, learning_rate=1e-3):
        self.input_layer = tf.placeholder(
            shape=[None, D_problem, D_problem, 2], dtype=tf.float32, name="X"
        )
        self.actual_actions = tf.placeholder(
            shape=[None, 4], dtype=tf.int32, name="actions"
        )
        self.game_rewards = tf.placeholder(shape=[None], dtype=tf.float32, name="rewards")
        # D = int(np.sqrt(D_problem))
        #
        # input_layer_reshaped = tf.reshape(self.input_layer, [-1, D, D, 2])
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=self.input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
        )
        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
        )
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        # Dense Layer
        pool2_flat = tf.reshape(
            pool2, [-1, pool2.shape[1] * pool2.shape[2] * pool2.shape[3]]
        )
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=False)
        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=n_classes)
        self.output_layer = tf.nn.softmax(logits)

        self.policy_loss = -1 * tf.losses.softmax_cross_entropy(
            onehot_labels=self.actual_actions,
            logits=logits,
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
