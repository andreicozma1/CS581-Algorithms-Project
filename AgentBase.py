import os
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


class AgentBase:
    def __init__(
        self,
        /,
        env="CliffWalking-v0",
        gamma=0.99,
        epsilon=0.1,
        run_name=None,
        seed=None,
        **kwargs,
    ):
        print("=" * 80)
        print(f"# Init Agent - {env}")

        self.env_name = env
        self.epsilon, self.gamma = float(epsilon), float(gamma)
        print(f"- epsilon: {self.epsilon}")
        print(f"- gamma: {self.gamma}")
        self.epsilon_override = None

        self.run_name = f"{run_name}_" if run_name is not None else ""
        self.run_name += f"{env}_gamma:{gamma}_epsilon:{epsilon}"
        print(f"- run_name: {run_name}")

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
            size = int(kwargs.get("size", 8))
            print(f"- size: {size}")
            self.run_name += f"_size:{size}"

            seed = int(seed) if seed is not None else np.random.randint(0, 100000)
            print(f"- seed: {seed}")
            self.run_name += f"_seed:{seed}"

            self.env_kwargs["desc"] = generate_random_map(size=size, seed=seed)
            self.env_kwargs["is_slippery"] = False

        self.env = gym.make(self.env_name, **self.env_kwargs)

        self.n_states, self.n_actions = (
            self.env.observation_space.n,
            self.env.action_space.n,
        )
        print(f"- n_states: {self.n_states}")
        print(f"- n_actions: {self.n_actions}")

    def choose_action(self, policy, state, greedy=False, **kwargs):
        """
        Sample an action from the policy.
        Also allows the ability to override the epsilon value (for the purpose of the demo)
        :param state: The current state
        :param policy: The policy to sample from. Must be of shape (n_states, n_actions)
        :param greedy: If True, always return the greedy action (argmax of the policy at the current state)
        :return: The sampled action
        """
        assert policy.shape == (self.n_states, self.n_actions), (
            f"ERROR: Policy must be of shape (n_states, n_actions) = ({self.n_states}, {self.n_actions}). "
            f"Got {policy.shape}."
        )

        # If greedy is True, always return the greedy action
        greedy_action = np.argmax(policy[state])
        if greedy or self.epsilon_override == 0.0:
            return greedy_action

        # Otherwise, sample an action from the soft policy (epsilon-greedy)
        if self.epsilon_override is None:
            return np.random.choice(self.n_actions, p=policy[state])

        # If we ever want to manually override the epsilon value, it happens here
        return np.random.choice(
            [greedy_action, np.random.randint(self.n_actions)],
            p=[1.0 - self.epsilon_override, self.epsilon_override],
        )

    def generate_episode(self, policy, max_steps=500, render=False, **kwargs):
        state, _ = self.env.reset()
        episode_hist, solved, done = [], False, False
        rgb_array = self.env.render() if render else None

        i = 0
        # Generate an episode following the current policy
        while i < max_steps and not solved and not done:
            # Render the environment if needed
            rgb_array = self.env.render() if render else None
            # Sample the next action from the policy
            action = self.choose_action(policy, state, **kwargs)
            # Keeping track of the trajectory
            episode_hist.append((state, action, None))
            # Take the action and observe the reward and next state
            next_state, reward, done, _, _ = self.env.step(action)
            if self.env_name == "FrozenLake-v1":
                if done:
                    reward = 100 if reward == 1 else -10
                else:
                    reward = -1

            # Keeping track of the trajectory
            episode_hist[-1] = (state, action, reward)
            # Generate the output at intermediate steps for the demo
            yield episode_hist, solved, rgb_array

            # For CliffWalking-v0 and Taxi-v3, the episode is solved when it terminates
            if done and self.env_name in ["CliffWalking-v0", "Taxi-v3"]:
                solved = True

            # For FrozenLake-v1, the episode terminates when the agent moves into a hole or reaches the goal
            # We consider the episode solved when the agent reaches the goal
            if done and self.env_name == "FrozenLake-v1":
                if next_state == self.env.nrow * self.env.ncol - 1:
                    solved = True
                else:
                    # Instead of terminating the episode when the agent moves into a hole, we reset the environment
                    # This is to keep consistent with the other environments
                    done, solved = False, False
                    next_state, _ = self.env.reset()

            state = next_state
            i += 1

        rgb_array = self.env.render() if render else None
        yield episode_hist, solved, rgb_array

    def run_episode(self, policy, max_steps=500, render=False, **kwargs):
        # Run the generator until the end
        episode_hist, solved, rgb_array = list(
            self.generate_episode(policy, max_steps, render, **kwargs)
        )[-1]
        return episode_hist, solved, rgb_array

    def test(self, n_test_episodes=100, verbose=True, greedy=True, **kwargs):
        if verbose:
            print(f"Testing agent for {n_test_episodes} episodes...")
        num_successes = 0
        for e in range(n_test_episodes):
            _, solved, _ = self.run_episode(policy=self.Pi, greedy=greedy, **kwargs)
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

    def save_policy(self, fname=None, save_dir=None):
        if fname is None and self.run_name is None:
            raise ValueError("Must provide a filename or a run name to save the policy")
        elif fname is None:
            fname = self.run_name

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, fname)

        if not fname.endswith(".npy"):
            fname += ".npy"

        print(f"Saving policy to: '{fname}'")
        np.save(fname, self.Pi)

    def load_policy(self, fname="policy.npy"):
        print(f"Loading policy from: '{fname}'")
        if not fname.endswith(".npy"):
            fname += ".npy"
        self.Pi = np.load(fname)
