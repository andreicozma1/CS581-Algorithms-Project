import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from matplotlib import pyplot as plt
from tqdm import trange
from Shared import Shared
import warnings


class DPAgent(Shared):
    def __init__(self,/,**kwargs):
        super().__init__(**kwargs)
        self.theta = kwargs.get('theta', 1e-10)
        print(self.theta)
        self.V = np.zeros(self.env.observation_space.n)
        self.Pi = np.zeros(self.env.observation_space.n, self.env.action_space.n)
        if self.gamma >= 1.0:
            warnings.warn("DP will never converge with a gamma value =1.0. Try 0.99?", UserWarning)

    def policy(self, state):
        return self.Pi[state]

    def train(self, *args, **kwargs):
        i = 0
        print(self.gamma)
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
            # if i % 100 == 0 and i != 0:
            #     self.test()
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
        idxs = np.argmax(self.Pi, axis=1)
        print(idxs)
        self.Pi = np.zeros((self.env.observation_space.n,self.env.action_space.n))
        self.Pi[np.arange(self.env.observation_space.n),idxs] = 1

        # print(self.Pi)
        # return self.V, self.Pi


if __name__ == "__main__":
    # env = gym.make('FrozenLake-v1', render_mode='human')
    dp = DPAgent(env_name="FrozenLake-v1")
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
        action = dp.choose_action(state)
        state, reward, done, _, _ = env.step(action)
        env.render()

    # plt.savefig(f"imgs/{0}.png")
