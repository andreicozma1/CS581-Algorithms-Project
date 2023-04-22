import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from matplotlib import pyplot as plt
from tqdm import trange


class DP:
    def __init__(self, env_name, gamma=0.9, theta=1e-10, **kwargs):
        self.env = gym.make(env_name, **kwargs)
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(self.env.observation_space.n)
        self.epsilon = 0

    def policy(self, state, return_value=False):
        Q = np.zeros(self.env.action_space.n)
        for action in range(self.env.action_space.n):
            expected_value = 0
            for probability, next_state, reward, done in self.env.P[state][action]:
                if state == self.env.observation_space.n-1: reward = 1
                expected_value += probability * (reward + self.gamma * self.V[next_state])
            Q[action] = expected_value
        if return_value:
            return np.argmax(Q), np.max(Q)
        else:
            return Q

    def train(self):
        i = 0
        while True:
            delta = 0
            V_prev = np.copy(self.V)
            for state in range(self.env.observation_space.n):
                action, value = self.policy(state, return_value=True)
                self.V[state] = value
                delta = max(delta, abs(V_prev[state] - self.V[state]))
            if delta < self.theta:
                break
            i += 1
            print(f"Iteration {i}: delta={delta}")
            # break

        policy = [self.policy(state, return_value=True)[0] for state in range(self.env.observation_space.n)]
        return self.V, policy
    
    def save_policy(self, pth):
        np.save(pth, self.V)

    def load_policy(self, pth):
        """
        not really loading the 'policy', but the state-value function but for
        interface's sake, here we are. 
        """
        self.V = np.load(pth)

    def generate_episode(self, max_steps, render=False, **kwargs):
        state, _ = self.env.reset()
        episode_hist, solved, rgb_array = [], False, None

        # Generate an episode following the current policy
        for _ in range(max_steps):
            rgb_array = self.env.render() if render else None
            # Sample an action from the policy
            action = self.policy(state)
            maction = np.argmax(action)
            # Take the action and observe the reward and next state
            next_state, reward, done, truncated, _ = self.env.step(maction)
            # Keeping track of the trajectory
            episode_hist.append((state, maction, reward))
            state = next_state

            yield episode_hist, solved, rgb_array

            # This is where the agent got to the goal.
            # In the case in which agent jumped off the cliff, it is simply respawned at the start position without termination.
            if done or truncated:
                solved = True
                break

        rgb_array = self.env.render() if render else None

        yield episode_hist, solved, rgb_array


if __name__ == "__main__":
    env = gym.make('CliffWalking-v0', render_mode='human')
    dp = DP(env)
    dp.train()
    dp.save_policy('dp_policy.npy')

    state, _ = env.reset()
    done = False
    while not done:
        action = dp.policy(state)
        state, reward, done, _, _ = env.step(action)
        env.render()

    # plt.savefig(f"imgs/{0}.png")
