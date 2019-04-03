"""
As the RL code will need a neural network this file is mean to implement
such network with the corresponding gradients.

For the moment I will use a NN with 1 hidden layer with relu activation
and 1 output layer using softmax.

Loss will be the cross entropy
"""
import tensorflow as tf
import numpy as np
import ipdb

np.random.seed(8)


def forward_pass(x):
    h1 = tf.nn.relu(x @ w1)
    y_pred = tf.nn.softmax(h1 @ w2)
    return y_pred, h1


def cross_entropy(y_pred, y_true):
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


N, D = 1, 2
D_h1 = 3  # Dimension first hidden layer
n_classes = 4
x_train = np.random.normal(size=(N, D))
y_train = np.array([[1.0, 0, 0, 0]])


w1_val = np.random.normal(size=(D, D_h1))
w2_val = np.random.normal(size=(D_h1, n_classes))

x = tf.placeholder(tf.float64, [N, D])
w1 = tf.get_variable("w1", initializer=tf.constant(w1_val))
w2 = tf.get_variable("w2", initializer=tf.constant(w2_val))

y_pred, h1 = forward_pass(x)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

y_pred_val = sess.run(y_pred, feed_dict={x: x_train})
loss = cross_entropy(y_pred, y_train)
loss_val = sess.run(loss, feed_dict={x: x_train})
derivative_w1 = tf.gradients(loss, w1)
ipdb.set_trace()
