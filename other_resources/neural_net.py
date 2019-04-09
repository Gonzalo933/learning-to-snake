import tensorflow as tf
import numpy as np
import ipdb

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K
from keras.initializers import Constant
from keras.losses import categorical_crossentropy

np.random.seed(8)


def relu(x):
    x[x < 0] = 0  # ReLU nonlinearity


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum()


def cross_entropy(y_net, y_train):
    return -1 * np.log(np.exp(y_net @ y_train.T) / np.sum(np.exp(y_net)))


def forward_pass_numpy(x):
    h1 = x @ w1
    relu(h1)
    h2 = h1 @ w2
    y = softmax(h2)
    return y, h1


N, D = 1, 2
D_h1 = 3  # Dimension first hidden layer
n_classes = 4
x = np.random.normal(size=(N, D))
w1 = np.random.normal(size=(D, D_h1))
w2 = np.random.normal(size=(D_h1, n_classes))

y, h1 = forward_pass_numpy(x)
y_train = np.array([[1.0, 0, 0, 0]])
loss = cross_entropy(y, y_train)

###########################
### Keras
###########################
model = Sequential()
model.add(
    Dense(
        D_h1,
        input_dim=D,
        kernel_initializer=Constant(value=w1),
        activation="relu",
        use_bias=False,
    )
)
model.add(
    Dense(
        n_classes,
        kernel_initializer=Constant(value=w2),
        activation="softmax",
        use_bias=False,
    )
)


def my_cross_entropy(y_true, y_pred):
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


model.compile(loss=my_cross_entropy, optimizer="adam", metrics=["accuracy"])

outputTensor = model.output  # Or model.layers[index].output
sess = tf.Session()
sess.run(tf.global_variables_initializer())
y_keras = sess.run(outputTensor, feed_dict={model.input: x})
loss_keras = K.eval(my_cross_entropy(K.variable(y_train), K.variable(y_keras)))
# model.fit(x, y_train, epochs=1, batch_size=2)
weights_keras = model.trainable_weights
gradients = K.gradients(outputTensor, weights_keras)
evaluated_gradients = sess.run(gradients, feed_dict={model.input: x})
ipdb.set_trace()

