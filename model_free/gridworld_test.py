import numpy as np
from gridworld import GridWorld

env = GridWorld(3, 4)

state_matrix = np.zeros((3,4))
state_matrix[0, 3] = 1
state_matrix[1, 3] = 1
state_matrix[1, 1] = -1

reward_matrix = np.full((3, 4), -0.04)
reward_matrix[0, 3] = 1
reward_matrix[1, 3] = -1

transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                              [0.1, 0.8, 0.1, 0.0],
                              [0.0, 0.1, 0.8, 0.1],
                              [0.1, 0.0, 0.1, 0.8]])

policy_matrix = np.array([[1, 1, 1, -1],
                          [0, np.NaN,  0,  -1],
                          [0,      3,  3,   3]])

env.setStateMatrix(state_matrix)
env.setRewardMatrix(reward_matrix)
env.setTransitionMatrix(transition_matrix)

observation = env.reset()

for _ in range(1000):
    action = policy_matrix[observation[0], observation[1]]
    observation, reward, done = env.step(action)
    print ""
    print "ACTION: " + str(action)
    print "REWARD: " + str(reward)
    print "DONE: " + str(done)
    env.render()
    if done: break
