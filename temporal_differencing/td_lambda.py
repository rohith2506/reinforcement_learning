'''
TD(Lambda) Learning

This contains eligiblity trace which cares about all previous states it has visited partially.

=> delta = reward(st+1) + gamma * U(st+1) - U(st)
=> U(st) = U(st) + delta * alpha * e(st)
=> e(st) => {
                lambda * gamma * e(st) + 1 (if s = st)
                lambda * gamma * e(st) ( if s != st)
            }

The advantage of this TD(lambda) is that it leaves the traces of its visited nature in all
the steps it has visited before

@Author: Rohith Uppala
'''

import numpy as np
from gridworld import GridWorld


def update_utility_matrix(utility_matrix, alpha, error, trace_matrix):
    utility_matrix += alpha * error * trace_matrix
    return utility_matrix

def update_eligibility_matrix(trace_matrix, gamma, lamda_):
    trace_matrix = trace_matrix * gamma * lamda_
    return trace_matrix

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
    trace_matrix = np.zeros((3, 4))


    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)

    utility_matrix = np.zeros((3, 4))
    gamma, alpha, tot_epoch, print_epoch, lambda_ = 0.999, 0.1, 30000, 1000, 0.5

    for epoch in range(tot_epoch):
        observation = env.reset(exploring_starts=True)
        for step in range(1000):
            action = policy_matrix[observation[0]][observation[1]]
            new_observation, reward, done = env.step(action)

            delta = reward + gamma * utility_matrix[new_observation[0]][new_observation[1]] - utility_matrix[observation[0]][observation[1]]
            trace_matrix[observation[0]][observation[1]] += 1

            utility_matrix = update_utility_matrix(utility_matrix, alpha, delta, trace_matrix)
            trace_matrix = update_eligibility_matrix(trace_matrix, gamma, lambda_)

            observation = new_observation
            if done: break
        if epoch % print_epoch == 0:
            print "utility matrix after %d iterations: " %(epoch)
            print utility_matrix

    print "final utility matrix: ", utility_matrix

if __name__ == "__main__":
    main()
