import gym
import gym_snake
from gym import envs

# print(envs.registry.all())
# Construct Environment
env = gym.make("snake-v0")

for i in range(100):
    env.reset()
    for t in range(1000):
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        import pdb

        pdb.set_trace()
        if done:
            print("episode {} finished after {} timesteps".format(i, t))
            break
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
