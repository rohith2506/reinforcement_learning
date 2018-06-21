import numpy as np
from gridworld import GridWorld

utility_matrix = np.zeros((3,4))
running_mean_matrix = np.full((3,4), 1.0e-10)
gamma, tot_epoch, print_epoch = 0.99, 5000, 1000

for epoch in range(tot_epoch):
    episode_list = list()
    observation = env.reset(exploring_starts=False)
    for _ in range(1000):
        action = policy_matrix[
