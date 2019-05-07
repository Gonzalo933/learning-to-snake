import gym
import gym_snake
from gym import envs
from utils import preprocess_board
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork, ConvolutionalNeuralNetwork
import tensorflow as tf
import os
import collections
import sys

# discount_rewards(np.array([1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0]))
# returns: array([ 1., 0.96059601, 0.970299, 0.9801, 0.99, 1., 0.9801, 0.99, 1.])
def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    food_obtained = (r == 1).sum()
    for t in reversed(range(0, r.size)):
        if r[t] == -1:
            # reset the sum, since this was a game boundary (pong specific!)
            running_add = 0
        elif r[t] == 1:
            food_obtained -= 1
            running_add = food_obtained
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


assert len(sys.argv) == 3, "Args needed"
use_convolutional = int(sys.argv[1]) == 1
print(f"Using convolutional? {use_convolutional}")
render = int(sys.argv[2]) == 1  # Show AI playing yes/no
restore_saved = False
gamma = 0.4  # Reward Discount multiplier
dim_hidden_layers = [12, 10, 5]
learning_rate = 1e-4
save_freq = 100  # keep zero if you dun want to save model
plot_freq = 500  # keep zero if you dun want to draw the scores
batch_size = 10  # every how many episodes to do a param update?
if use_convolutional:
    model_save_path = os.path.join(
        os.getcwd(), "model_tf_policyGrad_conv", "mymodel.ckpt"
    )
else:
    model_save_path = os.path.join(os.getcwd(), "model_tf_policyGrad", "mymodel.ckpt")
# print(envs.registry.all())
# Construct Environment
env = gym.make("snake-v0")
env.grid_size = [12, 12]
env.unit_gap = 0
env.random_init = False
frames_to_feed = 1
observation = env.reset()
if use_convolutional:
    downsampling = 1  # Set to 9 to set everything to 1 pixel wide.
else:
    downsampling = 1  # Set to 9 to set everything to 1 pixel wide.
last_frame = preprocess_board(observation, downsampling)
D = int(np.ceil(observation.shape[0] / downsampling) ** 2) * frames_to_feed
if use_convolutional:
    D = int((D / 2) ** 0.5)
n_classes = 4
reward_sum = 0
max_score = -1
Xs, dlogps, drs = [], [], []
last_scores = collections.deque(maxlen=50000)
last_means = collections.deque(maxlen=50000)
episode_number = 0
previous_x = None
tf.reset_default_graph()
if use_convolutional:
    nnet = ConvolutionalNeuralNetwork(D, n_classes, learning_rate=learning_rate)
else:
    nnet = NeuralNetwork(
        D, n_classes, dim_hidden_layers=dim_hidden_layers, learning_rate=learning_rate
    )
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_save_path))
    if ckpt and ckpt.model_checkpoint_path and restore_saved:
        print("using the saved model")
        nnet.saver.restore(sess, model_save_path)
    else:
        sess.run(tf.global_variables_initializer())
        print("Making new model")
    grad_buffer = sess.run(tf.trainable_variables())
    for indx, grad in enumerate(grad_buffer):
        grad_buffer[indx] = grad * 0
    while True:
        actual_frame = preprocess_board(observation, downsampling)
        # X = current_x - previous_x if previous_x is not None else np.zeros(D)
        # plt.figure(0)
        ## test = np.concatenate([last_frame, X])
        # plt.imshow(
        #    actual_frame.reshape(
        #        int(actual_frame.shape[0] ** 0.5), int(actual_frame.shape[0] ** 0.5)
        #    )
        # )
        # plt.show()
        # Forward pass and get action.
        if use_convolutional:
            X = np.stack(
                [last_frame.reshape((D, D)), actual_frame.reshape((D, D))], axis=2
            )
            X = X.reshape((1, X.shape[0], X.shape[1], X.shape[2]))
        else:
            # X = np.concatenate([last_frame, actual_frame])[None, :]
            X = actual_frame[None, :]
        actions_probs, = sess.run(nnet.output_layer, feed_dict={nnet.input_layer: X})
        last_frame = actual_frame
        # Sample action
        # actions_probs -> [UP, RIGHT, DOWN, LEFT]
        action = np.random.choice(4, p=actions_probs)
        observation, reward, done, info = env.step(action)  # env.action_space.sample()
        if render:
            env.render()
        # Set a label for the action taken as if it was the right action
        # because we still do not know if it is the right one.
        # print(f"Reward: {reward}")
        # print(f"Done: {done}")
        reward_sum += reward

        y = np.zeros(n_classes)
        y[action] = 1
        # grad that encourages the action that was taken to be taken
        dlogps.append(y)
        # add reward
        drs.append(float(reward))
        # add observation
        Xs.append(X)

        if done:
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(Xs)[:-1]
            epdlogp = np.vstack(dlogps)[:-1]
            epr = np.stack(drs)[:-1]
            Xs, dlogps, drs = [], [], []  # reset array memory
            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr, gamma=gamma)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            _ = sess.run(
                nnet.train_step,
                feed_dict={
                    nnet.input_layer: epx,
                    nnet.game_rewards: discounted_epr,
                    nnet.actual_actions: epdlogp,
                },
            )
            # grads = sess.run(
            #    nnet.all_gradients,
            #    feed_dict={
            #        nnet.input_layer: epx,
            #        nnet.game_rewards: discounted_epr,
            #        nnet.actual_actions: epdlogp,
            #    },
            # )
            # for indx, grad in enumerate(grads):
            #    grad_buffer[indx] += grad
            #
            ## perform rmsprop parameter update every batch_size episodes
            # if episode_number % batch_size == 0:
            #    print("Updating weights of the network")
            #    feed_dict = dict(zip(nnet.gradients, grad_buffer))
            #    _ = sess.run(nnet.apply_grads, feed_dict=feed_dict)
            #    for indx, grad in enumerate(grad_buffer):
            #        grad_buffer[indx] = grad * 0

            if save_freq and not (episode_number % save_freq):
                print("Saving the model ...")
                nnet.saver.save(sess, model_save_path)
            max_score = reward_sum if reward_sum > max_score else max_score
            running_mean = np.mean(list(last_scores))
            last_means.append(running_mean)
            if plot_freq and not (episode_number % plot_freq):
                # Tools > preferences > IPython console > Graphics > Graphics backend > Backend: Automatic
                # Then close and open Spyder.
                # plt.clf()
                fig = plt.figure(num=2)
                plt.plot(last_scores, ",")
                plt.plot(last_means, "-")
                plt.text(
                    0.01,
                    0.09,
                    f" num games: {episode_number} \n running mean: {running_mean:.3f} \n max score: {max_score}",
                    transform=plt.gcf().transFigure,
                )
                plt.subplots_adjust(left=0.3)
                plt.title("Tensorflow policy gradient")
                if use_convolutional:
                    plt.savefig("scores_summary_conv.png")
                else:
                    plt.savefig("scores_summary.png")
                plt.pause(0.0001)
                plt.close(fig)
            print(
                f"Resetting env. episode {episode_number} reward {reward_sum}. running mean: {running_mean:.3f} max score: {max_score}"
            )
            last_scores.append(reward_sum)
            reward_sum = 0
            observation = env.reset()  # reset env
            sys.stdout.flush()
# observation = env.reset()  # Constructs an instance of the game
# env.n_foods = 5
#
## Controller
# game_controller = env.controller
#
## Grid
# grid_object = game_controller.grid
# grid_pixels = grid_object.grid
#
## Snake(s)
# snakes_array = game_controller.snakes
# snake_object1 = snakes_array[0]
#
# observation = env.reset()
#
# env.render()
