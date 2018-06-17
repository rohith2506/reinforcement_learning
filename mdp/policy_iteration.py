"""
Implementation of policy iteration algorithm
@Author: Rohith Uppala
"""

import numpy as np

def return_policy_evaluation(p, u, r, T, gamma, total_states):
    for s in range(total_states):
        if not np.isnan(p[s]):
            v = np.zeros((1, total_states))
            v[0, s], action = 1.0, int(p[s])
            u[s] = r[s] + gamma * np.sum(np.multiply(u, np.dot(v, T[:, :, action])))
    return u

def return_expected_action(u, T, v):
    actions_array = np.zeros(4)
    for action in range(4):
        actions_array[action] = np.sum(np.multiply(u, np.dot(v, T[:, :, action])))
    return np.argmax(actions_array)

def print_policy(p, shape):
    counter, policy_string = 0, ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if p[counter] == -1: policy_string  += " *  "
            elif p[counter] == 0: policy_string += " ^  "
            elif p[counter] == 1: policy_string += " <  "
            elif p[counter] == 2: policy_string += " v  "
            elif p[counter] == 3: policy_string += " >  "
            elif np.isnan(p[counter]): policy_string += " #  "
            counter += 1
        policy_string += "\n"
    print policy_string

def main():
    gamma, epsilon, iteration, T, total_states = 0.999, 0.001, 0, np.load("T.npy"), 12
    p = np.random.randint(0, 4, size=(12)).astype(np.float32)
    p[5], p[3], p[7] = np.NaN, -1, -1
    u = np.zeros(total_states, dtype=float)
    r = np.array([-0.04, -0.04, -0.04,  +1.0,
                  -0.04,   0.0, -0.04,  -1.0,
                  -0.04, -0.04, -0.04, -0.04])
    while True:
        iteration += 1
        u_0, u = u.copy(), return_policy_evaluation(p, u, r, T, gamma, total_states)
        delta = np.max(np.abs(u - u_0))
        if delta < epsilon * (1 - gamma) / gamma: break
        for s in range(total_states):
            if not np.isnan(p[s]) and not p[s] == -1:
                v = np.zeros((1, total_states))
                v[0, s] = 1.0
                a = return_expected_action(u, T, v)
                if a != p[s]: p[s] = a
    print_policy(p, shape=(3,4))

if __name__ == "__main__":
    main()
