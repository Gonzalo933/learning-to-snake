import gym
from gym import envs
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import collections
from neural_network import NeuralNetwork
import sys


def discount_and_normalize_rewards(episode_rewards, gamma):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards


assert len(sys.argv) == 2, "Args needed"
render = int(sys.argv[1]) == 1  # Show AI playing yes/no
restore_saved = True
gamma = 0.95  # Reward Discount multiplier
dim_hidden_layers = [10, 5]
save_freq = 200  # keep zero if you dun want to save model
plot_freq = 2000  # keep zero if you dun want to draw the scores
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-3
model_save_path = os.path.join(os.getcwd(), "model_tf_policyGrad", "mymodel.ckpt")
# print(envs.registry.all())
# Construct Environment
env = gym.make("CartPole-v1")
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
observation = env.reset()
n_classes = action_space
reward_sum = 0
max_score = 0
means_to_plot = []
Xs, dlogps, drs = [], [], []
last_scores = collections.deque(maxlen=10000)
episode_number = 0
running_reward = None
tf.reset_default_graph()
nnet = NeuralNetwork(
    observation_space,
    n_classes,
    dim_hidden_layers=dim_hidden_layers,
    learning_rate=learning_rate,
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
        # X = current_x - previous_x if previous_x is not None else np.zeros(D)
        # plt.figure(0)
        # test = np.concatenate([last_frame, X])
        # plt.imshow(X.reshape((int(X.shape[0] ** 0.5), int(X.shape[0] ** 0.5))))
        # plt.show()
        # breakpoint()
        # Forward pass and get action.
        X = observation[None, :]
        actions_probs = sess.run(nnet.output_layer, feed_dict={nnet.input_layer: X})[0, 0]
        # if actions_probs >= 0.5 -> 1
        # Sample action
        # actions_probs -> [y == 0 =>  left] [y == 1 =>  right]
        action = np.random.choice(n_classes, p=[1 - actions_probs, actions_probs])
        observation, reward, done, info = env.step(action)  # env.action_space.sample()
        if render:
            env.render()
        # Set a label for the action taken as if it was the right action
        # because we still do not know if it is the right one.
        # print(f"Reward: {reward}")
        # print(f"Done: {done}")
        reward_sum += reward

        # log the action taken. Even though we don't know if the action is right
        # we take it as the true label for the observation.
        dlogps.append(action)
        # add reward for current action
        drs.append(float(reward))
        # log the observation
        Xs.append(X)
        if done:
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            # print("Updating weights of the network")
            epx = np.vstack(Xs)
            epdlogp = np.array(dlogps)[:, None]
            epr = np.stack(drs)
            Xs, dlogps, drs = [], [], []  # reset array memory
            # compute the discounted reward backwards through time
            discounted_epr = discount_and_normalize_rewards(epr, gamma=gamma)[:, None]
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
            # boring book-keeping
            max_score = reward_sum if reward_sum > max_score else max_score
            last_scores.append(reward_sum)
            running_mean = np.mean(list(last_scores)[-100:])
            means_to_plot.append(running_mean)
            solved = running_mean >= 195
            if plot_freq and not (episode_number % plot_freq):
                fig = plt.figure(num=2)
                plt.plot(last_scores, "-")
                plt.plot(means_to_plot, "-")
                plt.text(
                    0.01,
                    0.09,
                    f" num games: {episode_number} \n running mean: {running_mean:.3f} \n max score: {max_score}",
                    transform=plt.gcf().transFigure,
                )
                plt.subplots_adjust(left=0.3)
                plt.savefig("scores_summary.png")
                plt.title("Tensorflow policy gradient")
                plt.pause(0.0001)
                plt.close(fig)
            print(
                f"Resetting env. episode {episode_number} reward {reward_sum}. running mean: {running_mean:.3f} max score: {max_score}. Solved: {solved}"
            )
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
