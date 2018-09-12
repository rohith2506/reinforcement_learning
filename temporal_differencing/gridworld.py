import numpy as np

class GridWorld:
    def __init__(self, row,col):
        self.action_space_size = 4
        self.row = row
        self.col = col
        self.transition_matrix = np.ones((self.action_space_size, self.action_space_size)) /self.action_space_size
        self.reward_matrix = np.zeros((self.row, self.col))
        self.state_matrix = np.zeros((self.row, self.col))
        self.position = [np.random.randint(self.row), np.random.randint(self.col)]

    def setStateMatrix(self, state_matrix):
        if self.state_matrix.shape != state_matrix.shape:
            raise ValueError("Not Proper State Shape")
        self.state_matrix = state_matrix

    def setTransitionMatrix(self, transition_matrix):
        if self.transition_matrix.shape != transition_matrix.shape:
            raise ValueError("Not Proper Transition Shape")
        self.transition_matrix = transition_matrix

    def setRewardMatrix(self, reward_matrix):
        if self.reward_matrix.shape != reward_matrix.shape:
            raise ValueError("Not Proper Reward Shape")
        self.reward_matrix = reward_matrix

    def reset(self, exploring_starts=False):
        if exploring_starts:
            row, col = None, None
            while True:
                row = np.random.randint(0, self.row)
                col = np.random.randint(0, self.col)
                if self.state_matrix[row, col] == 0: break
            self.position = [row, col]
        else:
            self.position = [self.row-1, 0]
        return self.position

    def render(self):
        graph = ""
        for row in range(self.row):
            row_string = ""
            for col in range(self.col):
                if self.position == [row, col]:
                    row_string += u" \u25CB "
                else:
                    if self.state_matrix[row, col] == 0:  row_string += " - "
                    if self.state_matrix[row, col] == 1:  row_string += " # "
                    if self.state_matrix[row, col] == -1: row_string += " * "
            row_string += "\n"
            graph += row_string
        print(graph)

    def step(self, action):
        if action >= self.action_space_size:
            raise ValueError("The action is not included in action space")
        action = np.random.choice(self.action_space_size, 1, p=self.transition_matrix[int(action), :])
        if action == 0: new_position = [self.position[0]-1, self.position[1]]
        elif action == 1: new_position = [self.position[0], self.position[1]+1]
        elif action == 2: new_position = [self.position[0]+1, self.position[1]]
        elif action == 3: new_position = [self.position[0], self.position[1]-1]
        else: raise ValueError("The action is not included in action space")

        if new_position[0] >= 0 and new_position[0] < self.row:
            if new_position[1] >= 0 and new_position[1] < self.col:
                if self.state_matrix[new_position[0], new_position[1]] != -1:
                    self.position = new_position

        reward = self.reward_matrix[self.position[0], self.position[1]]
        done = bool(self.state_matrix[self.position[0], self.position[1]])
        return self.position, reward, done

