from CS231N_Project import the_agent
from CS231N_Project import environment
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import aircraft_TD
name = 'aircraft-TD-v0'

agent = the_agent.Agent(possible_actions=[0],starting_mem_len=50,max_mem_len=750000, starting_epsilon = .5, debug = True, learn_rate=0.0001)
env = environment.make_env(name,agent)

environment.play_episode(name, env,agent, debug = True)
env.close()

for i in range(0,len(agent.memory.frames)+1):
    fig = plt.figure(figsize = (7,7))
    state = [agent.memory.frames[i-3], agent.memory.frames[i-2], agent.memory.frames[i-1], agent.memory.frames[i]]
    for ind in range(4):
        state[ind] = [plt.imshow(state[ind], animated=True)]
    ani = animation.ArtistAnimation(fig, state, interval=750, blit=True,repeat_delay=250)

    plt.text(0, 0, 'Step: ' + str(i) + '    Reward: ' + str(agent.memory.rewards[i]) + '\nAction: ' + str(agent.memory.actions[i]) + "    Done: " + str(agent.memory.done_flags[i]) + '\n', fontsize=14, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
