"""
Implementation of Value Iteration Algorithm using Bellman Equation
@Author: Rohith Uppala
"""

import numpy as np
import pdb

def return_state_utility(v, T, u, reward, gamma):
    action_array = np.zeros(4)
    for action in range(0, 4):
        action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:, :, action])))
    return reward + gamma * np.max(action_array)

def main():
    total_states, gamma, iteration, epsilon = 12, 0.999, 0, 0.01
    graph_list = list()
    r = np.array([-0.04, -0.04, -0.04, +1.0,
                  -0.04, 0.0, -0.04, -1.0,
                  -0.04, -0.04, -0.04, -0.04])
    u = np.zeros(total_states, dtype=float)
    u1 = np.zeros(total_states, dtype=float)
    T = np.load("T.npy")
    while True:
        delta, u = 0, u1.copy()
        iteration += 1
        graph_list.append(u)
        for s in range(total_states):
            reward = r[s]
            v = np.zeros((1, total_states))
            v[0, s] = 1.0
            u1[s] = return_state_utility(v, T, u, reward, gamma)
            delta = max(delta, np.abs(u1[s] - u[s]))
        if delta < epsilon * (1 - gamma) / gamma:
            print "-------Final Result--------"
            print "Iterations: " + str(iteration)
            print "Delta: " + str(delta)
            print "Gamma: " + str(gamma)
            print "Epsilon: " + str(epsilon)
            print "---------------------------"
            print u
            break
    optimal_policy = np.zeros(total_states)
    for state in range(total_states):
        action_array = np.zeros(4)
        for action in range(0, 4):
            utility_sum = 0
            for next_state in range(total_states):
                utility_sum += np.multiply(T[state, next_state, action], u[next_state])
            action_array[action] = utility_sum
        optimal_policy[state] = np.argmax(action_array)
    print optimal_policy


if __name__ == "__main__":
    main()
