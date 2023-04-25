import warnings

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import trange

from AgentBase import AgentBase


class DPAgent(AgentBase):
    def __init__(self, /, **kwargs):
        super().__init__(run_name=self.__class__.__name__, **kwargs)
        self.theta = kwargs.get("theta", 1e-10)
        print(self.theta)
        self.V = np.zeros(self.env.observation_space.n)
        self.Pi = np.zeros(self.env.observation_space.n, self.env.action_space.n)
        if self.gamma >= 1.0:
            warnings.warn(
                "DP will never converge with a gamma value =1.0. Try 0.99?", UserWarning
            )

    def policy(self, state):
        return self.Pi[state]

    def train(self, *args, **kwargs):
        success_rate = []
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
                    for probability, next_state, reward, done in self.env.P[state][
                        action
                    ]:
                        if (
                            self.env_name == "CliffWalking-v0"
                            and state == self.env.observation_space.n - 1
                        ):
                            reward = 1
                        expected_value += probability * (
                            reward + self.gamma * self.V[next_state]
                        )
                    Q[action] = expected_value
                action, value = np.argmax(Q), np.max(Q)

                # update the state-value function
                self.V[state] = value
                delta = max(delta, abs(V_prev[state] - self.V[state]))
            self.make_pi()
            suc = self.test(verbose=False, greedy=True)
            success_rate.append(suc)
            if delta < self.theta and self.theta < 1:
                print(f"breaking at {delta}, {self.theta}")
                break
            elif i > self.theta and self.theta > 1:
                print(f"breaking at {i}, {self.theta}")
                break
            i += 1
            print(f"Iteration {i}: delta={delta}")

        # self.write_v(0)
        return success_rate

    def write_v(self, i):
        v_cop = np.copy(self.V).reshape((12, 4))
        print(v_cop)
        v_cop -= np.min(v_cop)
        v_cop /= np.max(v_cop)
        print(np.min(v_cop), np.max(v_cop))
        img = Image.fromarray(np.uint8(v_cop * 255), "L")
        img = img.resize(
            (v_cop.shape[0] * 100, v_cop.shape[1] * 100),
            resample=Image.Resampling.NEAREST,
        )
        img.save(f"imgs/{i}.png")

    def make_pi(self):
        self.Pi = np.empty((self.env.observation_space.n, self.env.action_space.n))
        for s in range(self.env.observation_space.n):
            for a in range(self.env.action_space.n):
                expected_value = 0
                for probability, next_state, reward, done in self.env.P[s][a]:
                    if (
                        self.env_name == "CliffWalking-v0"
                        and s == self.env.observation_space.n - 1
                    ):
                        reward = 1
                    expected_value += probability * (
                        reward + self.gamma * self.V[next_state]
                    )
                self.Pi[s, a] = expected_value
        idxs = np.argmax(self.Pi, axis=1)
        self.Pi = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.Pi[np.arange(self.env.observation_space.n), idxs] = 1


if __name__ == "__main__":
    env = gym.make(
        "FrozenLake-v1",
        render_mode="ansi",
        desc=generate_random_map(8, seed=24),
        is_slippery=False,
    )
    dp = DPAgent(env="FrozenLake-v1", gamma=0.99)
    dp.env = env
    dp.env_name = "FrozenLake-v1"
    dp.V = np.zeros(dp.env.observation_space.n)
    dp.Pi = np.zeros(dp.env.observation_space.n, dp.env.action_space.n)
    dp.n_states, dp.n_actions = (
        dp.env.observation_space.n,
        dp.env.action_space.n,
    )
    dp.train()

    print(dp.test())

    state, _ = env.reset()
    done = False
    while not done:
        action = dp.choose_action(state, greedy=True)
        state, reward, done, _, _ = env.step(action)
        s = env.render()
        print(s)
    plt.savefig(f"imgs/{0}.png")
