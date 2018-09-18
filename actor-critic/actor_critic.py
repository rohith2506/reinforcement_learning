'''
Actor-Critic Model

This is the ensemble of critic models such as (Monte Carlo, Temporal Differencing, Dynamic Programming) and actor models such as (Q-Learning, Reinforce)

@Author: Rohith Uppala
'''

import numpy as np
from gridworld import GridWorld

def softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def update_critic(utility_matrix, alpha, observation, new_observation, reward, gamma):
    u = utility_matrix[observation[0]][observation[1]]
    u_t1 = utility_matrix[new_observation[0]][new_observation[1]]
    delta = reward + gamma * u_t1 - u
    utility_matrix[observation[0]][observation[1]] += delta
    return utility_matrix, delta

def update_actor(state_action_matrix, observation, action, delta, beta_matrix=None):
    col = observation[1] + (4 * observation[0])
    if beta_matrix is None: beta = 1
    else: beta = 1 / beta_matrix[action, col]
    state_action_matrix[action, col] +=  beta * delta
    return state_action_matrix

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
    state_action_matrix = np.random.random((4, 12))
    print "State Action matrix"
    print state_action_matrix

    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)

    utility_matrix = np.zeros((3, 4))
    print "utility matrix"
    print utility_matrix

    gamma, alpha, tot_epoch, print_epoch = 0.999, 0.1, 30000, 1000

    for epoch in range(tot_epoch):
        observation = env.reset(exploring_starts=True)
        for step in range(1000):
            col = observation[1] + (4 * observation[0])

            # Sending Action to Environment
            action_array = state_action_matrix[:, col]
            action_distribution = softmax(action_array)
            action = np.random.choice(4, 1, p=action_distribution)

            new_observation, reward, done = env.step(action)

            # Update Critic
            utility_matrix, delta = update_critic(utility_matrix, alpha, observation, new_observation, reward, gamma)
            # Update Actor
            state_action_matrix = update_actor(state_action_matrix, observation, action, delta, beta_matrix=None)

            observation = new_observation
            if done: break

    print "final utility matrix"
    print utility_matrix
    print "final state action matrix"
    print state_action_matrix

if __name__ == "__main__":
    main()
