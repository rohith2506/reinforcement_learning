'''
Implementation of Monte Carlo Control Method

Equations:
    Q(s, a) = E{Return | st = s, at = a }
    π(s) = argmax(Q(s, a))

Author: Rohith Uppala

Note: We don't need any policy to determine the utility function
'''

import numpy as np
from gridworld import GridWorld

def get_return_value(state_list, gamma):
	counter, return_value = 0, 0
	for visit in state_list:
		return_value += visit[2] * np.power(gamma, counter)
		counter += 1
	return return_value

def update_policy_matrix(episode_list, policy_matrix, state_action_matrix):
	for visit in episode_list:
		observation = visit[0]
		col = observation[1] + (observation[0] * 4)
		if policy_matrix[observation[0], observation[1]] != -1:
			policy_matrix[observation[0], observation[1]] = np.argmax(state_action_matrix[:, col])
	return policy_matrix

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


	policy_matrix = np.random.randint(low=0, high=4, size=(3,4)).astype(np.float32)
	policy_matrix[1,1] = np.NaN
	policy_matrix[0,3] = policy_matrix[1, 3] = -1

	env.setStateMatrix(state_matrix)
	env.setRewardMatrix(reward_matrix)
	env.setTransitionMatrix(transition_matrix)


	state_action_matrix = np.random.random_sample((4, 12))
	running_mean_matrix = np.full((4, 12), 1.0e-10)
	gamma, total_epoch, print_epoch = 0.999, 50000, 3000

	for epoch in range(total_epoch):
		episode_list = list()
		observation = env.reset(exploring_starts=True)
		is_starting = True

		for _ in range(1000):
			action = policy_matrix[observation[0], observation[1]]
			if is_starting:
				action = np.random.randint(0, 4)
				is_starting = False
			new_observation, reward, done = env.step(action)
			episode_list.append((observation, action, reward))
			observation = new_observation
			if done: break

		counter = 0
		checkup_matrix = np.zeros((4, 12))
		for visit in episode_list:
			row, reward, col = visit[0][0], visit[1], visit[0][1] + (visit[0][0] * 4)
			if checkup_matrix[row, col] == 0:
				return_value = get_return_value(episode_list[counter:], gamma)
				running_mean_matrix[row, col] += 1
				state_action_matrix[row, col] += return_value
				checkup_matrix[row, col] = 1
			counter += 1

		policy_matrix = update_policy_matrix(episode_list, policy_matrix,
						state_action_matrix / running_mean_matrix)

		if epoch % print_epoch == 0:
			print(epoch)
			print("utility matrix: after ", epoch, " iterations")
			print(state_action_matrix / running_mean_matrix)
			print(policy_matrix)
			print()

	print("Final utility matix: ", (state_action_matrix / running_mean_matrix), "\n")
