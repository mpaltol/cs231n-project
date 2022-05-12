from IPython import display
import matplotlib.pyplot as plt

import gym
import aircraft_TD
env = gym.make('aircraft-TD-v0')

obs = env.reset()
plt.imshow(obs)
for i in range(1000):
    env.step(env.action_space.sample())
    env.render()