'''
TD(0) learning
Update rule: newEstimate <- oldEstimate + stepSize * ( target - oldEstimate )

@Author: Rohith Uppala
'''

import numpy as np
from gridworld import GridWorld

def update_utility_matrix(utility_matrix, observation, new_observation, gamma, alpha, reward):
    u1 = utility_matrix[observation[0]][observation[1]]
    u2 = utility_matrix[new_observation[0]][new_observation[1]]
    utility_matrix[observation[0]][observation[1]] += alpha * ( reward + gamma * u2 - u1 )
    return utility_matrix

def main():
    env = GridWorld(3, 4)
    state_matrix = np.zeros((3, 4))
    state_matrix[0, 3] = 1
    state_matrix[1, 3] = 1
    state_matrix[1, 1] = -1
    print state_matrix

    reward_matrix = np.full((3,4), -0.04)
    reward_matrix[0, 3] = 1
    reward_matrix[1, 3] = -1
    print reward_matrix

    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1], [0.1, 0.8, 0.1, 0.0], [0.0, 0.1, 0.8, 0.1], [0.1, 0.0, 0.1, 0.8]])
    policy_matrix = np.array([[1, 1, 1, -1], [0, np.NaN, 0, -1], [0, 3, 3, 3]])

    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)

    utility_matrix = np.zeros((3, 4))
    gamma, alpha, tot_epoch, print_epoch = 0.999, 0.1, 30000, 1000

    for epoch in range(tot_epoch):
        observation = env.reset(exploring_starts=False)
        for step in range(1000):
            action = policy_matrix[observation[0]][observation[1]]
            new_observation, reward, done = env.step(action)
            utility_matrix = update_utility_matrix(utility_matrix, observation, new_observation, gamma, alpha, reward)
            observation = new_observation
            if done: break
        if epoch % print_epoch == 0:
            print "utility matrix after %d iterations: " %(epoch)
            print utility_matrix

    print "final utility matrix: ", utility_matrix

if __name__ == "__main__":
    main()
