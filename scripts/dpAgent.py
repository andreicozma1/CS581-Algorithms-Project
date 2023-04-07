import numpy as np
import enum
from matplotlib import pyplot as plt
from tqdm import trange
from numba import njit, prange

from stationaryGrid import StationaryGrid

class Action(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class DP:
    def __init__(self, grid):
        self.grid = grid
        self.size = len(grid)
        self.V = np.zeros((self.size, self.size))
        self.gamma = 0.9
        self.actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    def rewardFunc(self, state, action):
        if action == Action.UP:
            finalPos = (state[0] - 1, state[1])
        elif action == Action.DOWN:
            finalPos = (state[0] + 1, state[1])
        elif action == Action.LEFT:
            finalPos = (state[0], state[1] - 1)
        elif action == Action.RIGHT:
            finalPos = (state[0], state[1] + 1)
        else:
            raise ValueError("Invalid action")

        if finalPos[0] < 0 or finalPos[0] >= self.size or finalPos[1] < 0 or finalPos[1] >= self.size:
            return state, -1
        elif self.grid[finalPos[0], finalPos[1]] == 0:
            return state, -1
        elif finalPos[0] == 0 and finalPos[1] == 0:
            return finalPos, 10

        return finalPos, 0

    # @njit(parallel=True)
    def run(self, num_iterations):
        for it in trange(num_iterations):
            V_copy = np.copy(self.V)
            for state in np.ndindex(*self.grid.shape):
                weighted_rewards = 0
                for action in self.actions:
                    finalPosition, reward = self.rewardFunc(state, action)
                    weighted_rewards += (1 / len(self.actions)) * (
                                reward + (self.gamma * self.V[finalPosition[0], finalPosition[1]]))
                V_copy[state[0], state[1]] = weighted_rewards
            self.V = V_copy

            # plt.imshow(self.V)
            # plt.savefig(f'imgs/{it}.png')
            # print(it)

    def policy(self, state):
        """
        The DP policy is to take the action that maximizes the value function.
        This returns the best action and the final position after taking that action.
        """
        r = -np.inf
        best = None
        bestPos = None
        for action in self.actions:
            finalPosition, reward = self.rewardFunc(state, action)
            if reward > r:
                r = reward
                best = action
                bestPos = finalPosition
        return best, bestPos, r

    def find_path(self):
        path = []
        cur = (self.size - 1, self.size - 1)
        path.append(cur)
        i = 0
        while cur != (0, 0):
            _, cur, _ = self.policy(cur)
            if cur in path:
                print(path, cur)
                raise ValueError("Infinite loop")
            path.append(cur)
        print(path)


if __name__ == "__main__":
    grid = StationaryGrid(4, size=20)
    grid.create_grid()
    dp = DP(grid.grid)
    dp.run(10000)
    dp.find_path()
    plt.imshow(dp.V)
    plt.savefig(f'imgs/{0}.png')
