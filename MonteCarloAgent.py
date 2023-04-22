import os
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import argparse
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import wandb


class MonteCarloAgent:
    def __init__(
        self,
        env_name="CliffWalking-v0",
        gamma=0.99,
        epsilon=0.1,
        run_name=None,
        **kwargs,
    ):
        print("=" * 80)
        print(f"# MonteCarloAgent - {env_name}")
        print(f"- epsilon: {epsilon}")
        print(f"- gamma: {gamma}")
        print(f"- run_name: {run_name}")
        self.run_name = run_name
        self.env_name = env_name
        self.epsilon, self.gamma = epsilon, gamma

        self.env_kwargs = kwargs
        if self.env_name == "FrozenLake-v1":
            self.env_kwargs["desc"] = None
            self.env_kwargs["map_name"] = "4x4"
            self.env_kwargs["is_slippery"] = "False"

        self.env = gym.make(self.env_name, **self.env_kwargs)

        self.n_states, self.n_actions = (
            self.env.observation_space.n,
            self.env.action_space.n,
        )
        print(f"- n_states: {self.n_states}")
        print(f"- n_actions: {self.n_actions}")
        self.reset()

    def reset(self):
        print("Resetting all state variables...")
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.R = [[[] for _ in range(self.n_actions)] for _ in range(self.n_states)]

        # An arbitrary e-greedy policy
        self.Pi = np.full(
            (self.n_states, self.n_actions), self.epsilon / self.n_actions
        )
        self.Pi[
            np.arange(self.n_states),
            np.random.randint(self.n_actions, size=self.n_states),
        ] = (
            1 - self.epsilon + self.epsilon / self.n_actions
        )
        print("=" * 80)
        print("Initial policy:")
        print(self.Pi)
        print("=" * 80)

    def choose_action(self, state, epsilon_override=None, greedy=False, **kwargs):
        # Sample an action from the policy.
        # The override_epsilon argument allows forcing the use of a possibly new self.epsilon value than the one used during training.
        # The ability to override was mostly added for testing purposes and for the demo.
        greedy_action = np.argmax(self.Pi[state])

        if greedy:
            return greedy_action

        if epsilon_override is None:
            return np.random.choice(self.n_actions, p=self.Pi[state])

        return np.random.choice(
            [greedy_action, np.random.randint(self.n_actions)],
            p=[1 - epsilon_override, epsilon_override],
        )

    def generate_episode(self, max_steps=500, render=False, **kwargs):
        state, _ = self.env.reset()
        episode_hist, solved, rgb_array = [], False, None

        # Generate an episode following the current policy
        while len(episode_hist) < max_steps:
            rgb_array = self.env.render() if render else None

            # Sample an action from the policy
            action = self.choose_action(state, **kwargs)
            # Take the action and observe the reward and next state
            next_state, reward, done, truncated, _ = self.env.step(action)

            # Keeping track of the trajectory
            episode_hist.append((state, action, reward))
            yield episode_hist, solved, rgb_array

            # For CliffWalking-v0 and Taxi-v3, the episode is solved when it terminates
            if done and (
                self.env_name == "CliffWalking-v0" or self.env_name == "Taxi-v3"
            ):
                solved = True
                break

            # For FrozenLake-v1, the episode terminates when the agent moves into a hole or reaches the goal
            # We consider the episode solved when the agent reaches the goal (done == True and reward == 1)
            if done and self.env_name == "FrozenLake-v1" and reward == 1:
                solved = True
                break

            if done or truncated:
                break

            state = next_state

        rgb_array = self.env.render() if render else None

        yield episode_hist, solved, rgb_array

    def run_episode(self, max_steps=500, render=False, **kwargs):
        # Run the generator until the end
        episode_hist, solved, rgb_array = None, False, None
        for episode_hist, solved, rgb_array in self.generate_episode(
            max_steps, render, **kwargs
        ):
            pass
        return episode_hist, solved, rgb_array

    def update_first_visit(self, episode_hist):
        G = 0
        # For each step of the episode, in reverse order
        for t in range(len(episode_hist) - 1, -1, -1):
            state, action, reward = episode_hist[t]
            # Update the expected return
            G = self.gamma * G + reward
            # If we haven't already visited this state-action pair up to this point, then we can update the Q-table and policy
            # This is the first-visit MC method
            if (state, action) not in [(x[0], x[1]) for x in episode_hist[:t]]:
                self.R[state][action].append(G)
                self.Q[state, action] = np.mean(self.R[state][action])
                # Epsilon-greedy policy update
                self.Pi[state] = np.full(self.n_actions, self.epsilon / self.n_actions)
                # the greedy action is the one with the highest Q-value
                self.Pi[state, np.argmax(self.Q[state])] = (
                    1 - self.epsilon + self.epsilon / self.n_actions
                )

    def update_every_visit(self, episode_hist):
        G = 0
        # For each step of the episode, in reverse order
        for t in range(len(episode_hist) - 1, -1, -1):
            state, action, reward = episode_hist[t]
            # Update the expected return
            G = self.gamma * G + reward
            # We update the Q-table and policy even if we have visited this state-action pair before
            # This is the every-visit MC method
            self.R[state][action].append(G)
            self.Q[state, action] = np.mean(self.R[state][action])
            # Epsilon-greedy policy update
            self.Pi[state] = np.full(self.n_actions, self.epsilon / self.n_actions)
            # the greedy action is the one with the highest Q-value
            self.Pi[state, np.argmax(self.Q[state])] = (
                1 - self.epsilon + self.epsilon / self.n_actions
            )

    def train(
        self,
        n_train_episodes=2000,
        test_every=100,
        update_type="first_visit",
        log_wandb=False,
        save_best=True,
        save_best_dir=None,
        **kwargs,
    ):
        print(f"Training agent for {n_train_episodes} episodes...")

        (
            train_running_success_rate,
            test_success_rate,
            test_running_success_rate,
            avg_ep_len,
        ) = (0.0, 0.0, 0.0, 0.0)

        stats = {
            "train_running_success_rate": train_running_success_rate,
            "test_running_success_rate": test_running_success_rate,
            "test_success_rate": test_success_rate,
            "avg_ep_len": avg_ep_len,
        }

        update_func = getattr(self, f"update_{update_type}")

        tqrange = tqdm(range(n_train_episodes))
        tqrange.set_description("Training")

        if log_wandb:
            self.wandb_log_img(episode=None)

        for e in tqrange:
            episode_hist, solved, _ = self.run_episode(**kwargs)
            rewards = [x[2] for x in episode_hist]
            total_reward, avg_reward = sum(rewards), np.mean(rewards)

            train_running_success_rate = (
                0.99 * train_running_success_rate + 0.01 * solved
            )
            avg_ep_len = 0.99 * avg_ep_len + 0.01 * len(episode_hist)

            update_func(episode_hist)

            stats = {
                "train_running_success_rate": train_running_success_rate,
                "test_running_success_rate": test_running_success_rate,
                "test_success_rate": test_success_rate,
                "avg_ep_len": avg_ep_len,
                "total_reward": total_reward,
                "avg_reward": avg_reward,
            }
            tqrange.set_postfix(stats)

            # Test the agent every test_every episodes with the greedy policy (by default)
            if e % test_every == 0:
                test_success_rate = self.test(verbose=False, **kwargs)
                if save_best and test_success_rate > 0.9:
                    if self.run_name is None:
                        print(f"Warning: run_name is None, not saving best policy")
                    else:
                        self.save_policy(self.run_name, save_best_dir)

                if log_wandb:
                    self.wandb_log_img(episode=e)

            test_running_success_rate = (
                0.99 * test_running_success_rate + 0.01 * test_success_rate
            )
            stats["test_running_success_rate"] = test_running_success_rate
            stats["test_success_rate"] = test_success_rate
            tqrange.set_postfix(stats)

            if log_wandb:
                wandb.log(stats)

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

    def wandb_log_img(self, episode=None):
        caption_suffix = "Initial" if episode is None else f"After Episode {episode}"
        wandb.log(
            {
                "Q-table": wandb.Image(
                    self.Q,
                    caption=f"Q-table - {caption_suffix}",
                ),
                "Policy": wandb.Image(
                    self.Pi,
                    caption=f"Policy - {caption_suffix}",
                ),
            }
        )


def main():
    parser = argparse.ArgumentParser()

    ### Train/Test parameters
    parser.add_argument(
        "--train",
        action="store_true",
        help="Use this flag to train the agent.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Use this flag to test the agent. Provide the path to the policy file.",
    )
    parser.add_argument(
        "--n_train_episodes",
        type=int,
        default=2500,
        help="The number of episodes to train for. (default: 2500)",
    )
    parser.add_argument(
        "--n_test_episodes",
        type=int,
        default=100,
        help="The number of episodes to test for. (default: 100)",
    )
    parser.add_argument(
        "--test_every",
        type=int,
        default=100,
        help="During training, test the agent every n episodes. (default: 100)",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="The maximum number of steps per episode before the episode is forced to end. (default: 200)",
    )

    parser.add_argument(
        "--update_type",
        type=str,
        choices=["first_visit", "every_visit"],
        default="first_visit",
        help="The type of update to use. (default: first_visit)",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="policies",
        help="The directory to save the policy to. (default: policies)",
    )

    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Use this flag to disable saving the policy.",
    )

    ### Agent parameters
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="The value for the discount factor to use. (default: 1.0)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.4,
        help="The value for the epsilon-greedy policy to use. (default: 0.4)",
    )

    ### Environment parameters
    parser.add_argument(
        "--env",
        type=str,
        default="CliffWalking-v0",
        choices=["CliffWalking-v0", "FrozenLake-v1", "Taxi-v3"],
        help="The Gymnasium environment to use. (default: CliffWalking-v0)",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,
        help="Render mode passed to the gym.make() function. Use 'human' to render the environment. (default: None)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project name for logging. If not provided, no logging is done. (default: None)",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default="monte-carlo",
        help="WandB group name for logging. (default: monte-carlo)",
    )
    parser.add_argument(
        "--wandb_job_type",
        type=str,
        default="train",
        help="WandB job type for logging. (default: train)",
    )
    parser.add_argument(
        "--wandb_run_name_suffix",
        type=str,
        default=None,
        help="WandB run name suffix for logging. (default: None)",
    )

    args = parser.parse_args()

    agent = MonteCarloAgent(
        args.env,
        gamma=args.gamma,
        epsilon=args.epsilon,
        render_mode=args.render_mode,
    )

    run_name = f"{agent.__class__.__name__}_{args.env}_e{args.n_train_episodes}_s{args.max_steps}_g{args.gamma}_e{args.epsilon}_{args.update_type}"
    if args.wandb_run_name_suffix is not None:
        run_name += f"+{args.wandb_run_name_suffix}"

    agent.run_name = run_name

    try:
        if args.train:
            # Log to WandB
            if args.wandb_project is not None:
                wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    group=args.wandb_group,
                    job_type=args.wandb_job_type,
                    config=dict(args._get_kwargs()),
                )

            agent.train(
                n_train_episodes=args.n_train_episodes,
                test_every=args.test_every,
                n_test_episodes=args.n_test_episodes,
                max_steps=args.max_steps,
                update_type=args.update_type,
                log_wandb=args.wandb_project is not None,
                save_best=True,
                save_best_dir=args.save_dir,
            )
            if not args.no_save:
                agent.save_policy(
                    fname=f"{run_name}.npy",
                    save_dir=args.save_dir,
                )
        elif args.test is not None:
            if not args.test.endswith(".npy"):
                args.test += ".npy"
            agent.load_policy(args.test)
            agent.test(
                n_test_episodes=args.n_test_episodes,
                max_steps=args.max_steps,
            )
        else:
            print("ERROR: Please provide either --train or --test.")
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()
