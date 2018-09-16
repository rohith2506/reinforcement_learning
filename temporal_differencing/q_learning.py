'''
Q - learning ( Known as Temporal differencing with control using an off-policy mechanism)

Here, we learn through observation. No updates of policy matrix while trying to acheive
the convergence

Q(st, at) <- Q(st, at) + alpha * ( r(t+1)  + gamma * (max(q(st+1, at+1))) - q(st, at) )

@Author: Rohith Uppala
'''

import numpy as np
from gridworld import GridWorld

def update_state_action_matrix(state_action_matrix, reward, gamma, observation, new_observation, action, visit_counter_matrix):
	col = observation[1] + (4 * observation[0])
	col_t1 = new_observation[1] + (4 * new_observation[0])

        q = state_action_matrix[action, col]
        # This is the only difference between SARSA and Q - Learning instead of incremental learning
        # We usually go with the sub-optimal policy
        q_t1 = np.max(state_action_matrix[:, col_t1])

        # Instead of going for a constant, let's calculate based on how many times we have visited this state
	alpha = 1 / (1.0 + visit_counter_matrix[action, col])
        state_action_matrix[action, col] = state_action_matrix[action, col] + alpha * ( reward + gamma * q_t1 - q)
        return state_action_matrix


def update_policy(policy_matrix, state_action_matrix, observation):
	col = observation[1] + (observation[0] * 4)
	best_action = np.argmax(state_action_matrix[:, col])
	policy_matrix[observation[0]][observation[1]] = best_action
	return policy_matrix

def update_visit_counter(visit_counter_matrix, observation, action):
	col = observation[1] + (observation[0] * 4)
	visit_counter_matrix[action, col] += 1
	return visit_counter_matrix

def return_epsilon_greedy_action(policy_matrix, observation, epsilon=0.1):
	tot_actions = int(np.nanmax(policy_matrix) + 1)
        action = int(policy_matrix[observation[0]][observation[1]])
        non_greedy_prob = epsilon / tot_actions
	greedy_prob = 1 - epsilon  + non_greedy_prob
	weight_array = np.full((tot_actions), non_greedy_prob)
	weight_array[action] = greedy_prob
	return np.random.choice(tot_actions, 1, p=weight_array)

def print_policy(policy_matrix):
	counter = 0
	shape = policy_matrix.shape
	policy_string = ""
	for row in range(shape[0]):
		for col in range(shape[1]):
			if(policy_matrix[row,col] == -1): policy_string += " *  "
			elif(policy_matrix[row,col] == 0): policy_string += " ^  "
			elif(policy_matrix[row,col] == 1): policy_string += " >  "
			elif(policy_matrix[row,col] == 2): policy_string += " v  "
			elif(policy_matrix[row,col] == 3): policy_string += " <  "
			elif(np.isnan(policy_matrix[row,col])): policy_string += " #  "
			counter += 1
		policy_string += '\n'
	print(policy_string)

def return_decayed_value(starting_value, global_step, decay_step):
	return starting_value * np.power(0.1, (global_step/decay_step))

def main():
	env = GridWorld(3, 4)
	state_matrix = np.zeros((3, 4))
	state_matrix[0, 3] = 1
	state_matrix[1, 3] = 1
	state_matrix[1, 1] = -1

	reward_matrix = np.full((3,4), -0.04)
	reward_matrix[0, 3] = 1
	reward_matrix[1, 3] = -1

	transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1], [0.1, 0.8, 0.1, 0.0], [0.0, 0.1, 0.8, 0.1], [0.1, 0.0, 0.1, 0.8]])

        # Generating random policy
        policy_matrix = np.random.randint(low=0, high=4, size=(3, 4)).astype(np.float32)
        policy_matrix[1,1] = np.NaN
        policy_matrix[0, 3] = policy_matrix[1, 3] = -1
        print "Random policy matrix"
        print_policy(policy_matrix)

        exploratory_policy_matrix = np.array([[1, 1, 1, -1], [0, np.NaN, 0, -1], [0, 1, 0, 3]])
        print "Exploratory policy matrix"
        print_policy(exploratory_policy_matrix)

	env.setStateMatrix(state_matrix)
	env.setRewardMatrix(reward_matrix)
	env.setTransitionMatrix(transition_matrix)

	state_action_matrix = np.zeros((4,12))
	visit_counter_matrix = np.zeros((4, 12))

	utility_matrix = np.zeros((3, 4))
	gamma, alpha, tot_epoch, print_epoch = 0.999, 0.001, 500000, 1000

	for epoch in range(tot_epoch):
		epsilon = return_decayed_value(0.1, epoch, decay_step=100000)
		observation = env.reset(exploring_starts=True)
		is_starting = True

		for step in range(1000):
			action = return_epsilon_greedy_action(exploratory_policy_matrix, observation, epsilon=0.1)
			if is_starting:
				action = np.random.randint(0, 4)
				is_starting = False

			new_observation, reward, done = env.step(action)

			state_action_matrix = update_state_action_matrix(state_action_matrix, reward, gamma, observation, new_observation, action, visit_counter_matrix)
			policy_matrix = update_policy(policy_matrix, state_action_matrix, observation)
			visit_counter_matrix = update_visit_counter(visit_counter_matrix, observation, action)

			observation = new_observation
			if done: break

		if epoch % print_epoch == 0:
			print "state action and policy matrices after %d iterations: " %(epoch)
		        print state_action_matrix
                        print "best policy after %d iterations: " %(epoch)
                        print_policy(policy_matrix)
                        print "##################################"

	print "final state action matrix: ", state_action_matrix
	print "final policy matrix: ", policy_matrix

if __name__ == "__main__":
	main()
