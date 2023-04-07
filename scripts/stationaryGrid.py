import numpy as np
import random
from matplotlib import pyplot as plt
from astar import AStar
import pickle
import enum

class MazeSolver(AStar):
    """
    Because I'm too lazy to implement A-star, this class yoinked
    from https://github.com/jrialland/python-astar/blob/f11311b678522d90c1786e6b8d9393095a0b733f/tests/maze/test_maze.py#L58

    Sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position
    """

    def __init__(self, maze):
        self.world = maze
        self.size = maze.shape[0]

    def heuristic_cost_estimate(self, n1, n2):
        """computes the 'direct' distance between two (x,y) tuples"""
        (x1, y1) = n1
        (x2, y2) = n2
        return np.hypot(x2 - x1, y2 - y1)

    def distance_between(self, n1, n2):
        """this method always returns 1, as two 'neighbors' are always adajcent"""
        return 1

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        x, y = node
        return[(nx, ny) for nx, ny in[(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)] if 0 <= nx < self.size and 0 <= ny < self.size and self.world[ny,nx] == 1]

class StationaryGrid:
    def __init__(self, seed, size=20):
        np.random.seed(seed)
        random.seed(seed)
        self.size = size
        self.grid = np.ones((size, size),dtype=np.uint8)

    def create_grid(self):
        n_obstacles = np.random.randint(1, 10)
        i = 0
        while i < n_obstacles:
        # for i in range(n_obstacles):
            x = np.random.randint(0, self.grid.shape[0])
            y = np.random.randint(0, self.grid.shape[1])
            size = np.random.randint(2, high=self.size // 2, size=(2,))
            if x == 0 and y == 0:
                continue
            if (x + size[0]) >= self.grid.shape[0] and (y + size[1]) >= self.grid.shape[1]:
                continue

            start = (0, 0)
            goal = (self.grid.shape[0] - 2, self.grid.shape[0] - 2)
            self.grid[x:x + size[0], y:y + size[1]] = 0
            # make sure there's still a path to the goal
            path = MazeSolver(self.grid).astar(start, goal)
            if path is None:
                # if not, undo the current obstacle and generate another random one
                self.grid[x:x + size[0], y:y + size[1]] = 1
                continue

            i += 1

    def plot(self, pth=None):
        plt.imshow(self.grid, cmap='gray')
        plt.plot(0, 0,marker='o', markersize=10, color="red")
        plt.plot(self.grid.shape[0]-1, self.grid.shape[1]-1, marker='o', markersize=10, color="green")
        if pth is not None:
            plt.savefig(pth)
        else:
            plt.show()


if __name__ == '__main__':
    for i in range(100):
        grid = StationaryGrid(i, size=100)
        grid.create_grid()
        grid.plot(f'imgs/{i}.png')
        print(i)
