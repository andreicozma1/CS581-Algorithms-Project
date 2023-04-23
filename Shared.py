import os
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


class Shared:
    def __init__(
        self,
        /,
        env="CliffWalking-v0",
        gamma=0.99,
        epsilon=0.1,
        run_name=None,
        frozenlake_size=8,
        **kwargs,
    ):
        print("=" * 80)
        print(f"# Init Agent - {env}")
        print(f"- epsilon: {epsilon}")
        print(f"- gamma: {gamma}")
        print(f"- run_name: {run_name}")
        self.run_name = run_name
        self.env_name = env
        self.epsilon, self.gamma = epsilon, gamma
        self.epsilon_override = None

        self.env_kwargs = {k: v for k, v in kwargs.items() if k in ["render_mode"]}
        if self.env_name == "FrozenLake-v1":
            # Can use defaults by defining map_name (4x4 or 8x8) or custom map by defining desc
            # self.env_kwargs["map_name"] = "8x8"
            # self.env_kwargs["desc"] = [
            #     "SFFFFFFF",
            #     "FFFFFFFH",
            #     "FFFHFFFF",
            #     "FFFFFHFF",
            #     "FFFHFFFF",
            #     "FHHFFFHF",
            #     "FHFFHFHF",
            #     "FFFHFFFG",
            # ]
            self.env_kwargs["desc"] = generate_random_map(size=frozenlake_size)
            self.env_kwargs["is_slippery"] = False

        self.env = gym.make(self.env_name, **self.env_kwargs)

        self.n_states, self.n_actions = (
            self.env.observation_space.n,
            self.env.action_space.n,
        )
        print(f"- n_states: {self.n_states}")
        print(f"- n_actions: {self.n_actions}")

    def choose_action(self, state, greedy=False, **kwargs):
        # Sample an action from the policy.
        # The epsilon_override argument allows forcing the use of a new epsilon value than the one previously used during training.
        # The ability to override was mostly added for testing purposes and for the demo.
        greedy_action = np.argmax(self.Pi[state])

        if greedy or self.epsilon_override == 0.0:
            return greedy_action

        if self.epsilon_override is None:
            return np.random.choice(self.n_actions, p=self.Pi[state])

        print("epsilon_override", self.epsilon_override)
        return np.random.choice(
            [greedy_action, np.random.randint(self.n_actions)],
            p=[1.0 - self.epsilon_override, self.epsilon_override],
        )

    def generate_episode(self, max_steps=500, render=False, **kwargs):
        state, _ = self.env.reset()
        episode_hist, solved, rgb_array = (
            [],
            False,
            self.env.render() if render else None,
        )

        # Generate an episode following the current policy
        for _ in range(max_steps):
            # Sample an action from the policy
            action = self.choose_action(state, **kwargs)
            # Take the action and observe the reward and next state
            next_state, reward, done, _, _ = self.env.step(action)

            if self.env_name == "FrozenLake-v1":
                if done:
                    reward = 100 if reward == 1 else -10
                else:
                    reward = -1

            # Keeping track of the trajectory
            episode_hist.append((state, action, reward))
            yield episode_hist, solved, rgb_array

            # Rendering new frame if needed
            rgb_array = self.env.render() if render else None

            # For CliffWalking-v0 and Taxi-v3, the episode is solved when it terminates
            if done and self.env_name in ["CliffWalking-v0", "Taxi-v3"]:
                solved = True
                break

            # For FrozenLake-v1, the episode terminates when the agent moves into a hole or reaches the goal
            # We consider the episode solved when the agent reaches the goal
            if done and self.env_name == "FrozenLake-v1":
                if next_state == self.env.nrow * self.env.ncol - 1:
                    solved = True
                    break
                else:
                    # Instead of terminating the episode when the agent moves into a hole, we reset the environment
                    # This is to keep consistent with the other environments
                    done = False
                    next_state, _ = self.env.reset()

            if solved or done:
                break

            state = next_state

        rgb_array = self.env.render() if render else None
        yield episode_hist, solved, rgb_array

    def run_episode(self, max_steps=500, render=False, **kwargs):
        # Run the generator until the end
        episode_hist, solved, rgb_array = list(
            self.generate_episode(max_steps, render, **kwargs)
        )[-1]
        return episode_hist, solved, rgb_array

    def test(self, n_test_episodes=100, verbose=True, greedy=True, **kwargs):
        if verbose:
            print(f"Testing agent for {n_test_episodes} episodes...")
        num_successes = 0
        for e in range(n_test_episodes):
            _, solved, _ = self.run_episode(greedy=greedy, **kwargs)
            num_successes += solved
            if verbose:
                word = "reached" if solved else "did not reach"
                emoji = "🏁" if solved else "🚫"
                print(
                    f"({e + 1:>{len(str(n_test_episodes))}}/{n_test_episodes}) - Agent {word} the goal {emoji}"
                )

        success_rate = num_successes / n_test_episodes
        if verbose:
            print(
                f"Agent reached the goal in {num_successes}/{n_test_episodes} episodes ({success_rate * 100:.2f}%)"
            )
        return success_rate

    def save_policy(self, fname="policy.npy", save_dir=None):
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, fname)
        print(f"Saving policy to: {fname}")
        np.save(fname, self.Pi)

    def load_policy(self, fname="policy.npy"):
        print(f"Loading policy from: {fname}")
        self.Pi = np.load(fname)
