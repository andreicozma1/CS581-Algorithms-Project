import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from matplotlib import pyplot as plt
from tqdm import trange
from Shared import Shared


class DPAgent(Shared):
    def __init__(self, theta=1e-10, **kwargs):
        super().__init__(**kwargs)
        self.theta = theta
        self.V = np.zeros(self.env.observation_space.n)
        self.Pi = None

    def policy(self, state):
        return self.Pi[state]

    def train(self, **kwargs):
        i = 0
        while True:
            delta = 0
            V_prev = np.copy(self.V)
            for state in range(self.env.observation_space.n):
                # calculate the action-value for each possible action
                Q = np.zeros(self.env.action_space.n)
                for action in range(self.env.action_space.n):
                    expected_value = 0
                    for probability, next_state, reward, done in self.env.P[state][action]:
                        if state == self.env.observation_space.n-1: reward = 1
                        expected_value += probability * (reward + self.gamma * self.V[next_state])
                    Q[action] = expected_value
                action, value = np.argmax(Q), np.max(Q)

                # update the state-value function
                self.V[state] = value
                delta = max(delta, abs(V_prev[state] - self.V[state]))
            if delta < self.theta:
                break
            i += 1
            self.test()
            print(f"Iteration {i}: delta={delta}")
            # break

        # policy = [self.policy(state, return_value=True)[0] for state in range(self.env.observation_space.n)]
        self.Pi = np.empty((self.env.observation_space.n, self.env.action_space.n))
        for s in range(self.env.observation_space.n):
            for a in range(self.env.action_space.n):
                expected_value = 0
                for probability, next_state, reward, done in self.env.P[s][a]:
                    # if state == self.env.observation_space.n-1: reward = 1
                    expected_value += probability * (reward + self.gamma * self.V[next_state])
                self.Pi[s,a] = expected_value
        self.Pi = np.argmax(self.Pi, axis=1)
        print(self.Pi)
        # return self.V, self.Pi


if __name__ == "__main__":
    # env = gym.make('FrozenLake-v1', render_mode='human')
    dp = DPAgent("FrozenLake-v1", is_slippery=False, desc=[
        "SFFFFFFF",
        "FFFFFFFH",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ])
    dp.train()
    dp.save_policy('dp_policy.npy')
    env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=False, desc=[
        "SFFFFFFF",
        "FFFFFFFH",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ])

    state, _ = env.reset()
    done = False
    while not done:
        action = dp.policy(state)
        action = np.argmax(action)
        state, reward, done, _, _ = env.step(action)
        env.render()

    # plt.savefig(f"imgs/{0}.png")
