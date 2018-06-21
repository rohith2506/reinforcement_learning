'''
Implementation of monte carlo predict model
Given an optimal policy, It will predict the utility function

@Author: Rohith Uppala
'''

import numpy as np
from gridworld import GridWorld


def get_return_value(state_list, gamma):
	counter, return_value = 0, 0
	for visit in state_list:
		return_value += visit[1] * np.power(gamma, counter)
		counter += 1
	return return_value

if __name__ == "__main__":
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

	utility_matrix = np.zeros((3, 4))
	running_mean_matrix = np.full((3, 4), 1.0e-10)
	gamma, total_epoch, print_epoch = 0.999, 5000, 1000

	for epoch in range(total_epoch):
		episode_list = list()
		observation = env.reset(exploring_starts=True)
		for _ in range(print_epoch):
			action = policy_matrix[observation[0], observation[1]]
			observation, reward, done = env.step(action)
			episode_list.append((observation, reward))
			if done: break
		counter = 0
		checkup_matrix = np.zeros((3, 4))
		for visit in episode_list:
			row, col, reward = visit[0][0], visit[0][1], visit[1]
			if checkup_matrix[row, col] == 0:
				return_value = get_return_value(episode_list[counter:], gamma)
				running_mean_matrix[row, col] += 1
				utility_matrix[row, col] += return_value
				checkup_matrix[row, col] = 1
			counter += 1
		if epoch % print_epoch == 0:
			print("utility matrix: after ", print_epoch, " iterations")
			print(utility_matrix / running_mean_matrix)

	print("Final utility matix: ", (utility_matrix / running_mean_matrix), "\n")
