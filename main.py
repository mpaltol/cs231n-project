from CS231N_Project import the_agent
from CS231N_Project import environment
import matplotlib.pyplot as plt
import time
from collections import deque
import numpy as np
import aircraft_TD

name = 'aircraft-TD-v0'

agent = the_agent.Agent(possible_actions=[0,1,3],starting_mem_len=20000,max_mem_len=750000,starting_epsilon = 1, learn_rate = .0025)
env = environment.make_env(name,agent)

last_100_avg = [-21]
scores = deque(maxlen = 1100)
max_score = -21

""" If testing:
agent.model.load_weights('recent_weights.hdf5')
agent.model_target.load_weights('recent_weights.hdf5')
agent.epsilon = 0.0
"""

env.reset()

for i in range(1100):
    timesteps = agent.total_timesteps
    timee = time.time()
    score = environment.play_episode(name, env, agent, debug = True) #set debug to true for rendering
    scores.append(score)
    if score > max_score:
        max_score = score

    print('\nEpisode: ' + str(i))
    print('Steps: ' + str(agent.total_timesteps - timesteps))
    print('Duration: ' + str(time.time() - timee))
    print('Score: ' + str(score))
    print('Max Score: ' + str(max_score))
    print('Epsilon: ' + str(agent.epsilon))
    #if i%10==0 and i!=0:
    #    env.render()
    interval = 200
    if i%interval==0 and i!=0:
        #last_100_avg.append(sum(scores)/len(scores))
        #plt.plot(np.arange(0,i+1,interval),last_100_avg)
        ascores = np.array(scores)
        plt.plot(np.arange(0, i+1), 0.01*ascores)
        plt.xlabel('Episode number')
        plt.ylabel('Score')
        np.savetxt("eps1_95_20000.csv", ascores, delimiter=",")
        plt.show()
        
