import numpy as np
from tqdm import tqdm
from Shared import Shared
import wandb
from Shared import Shared


class MCAgent(Shared):
    def __init__(self, /, **kwargs):
        super().__init__(run_name=self.__class__.__name__, **kwargs)
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
        early_stopping=False,
        **kwargs,
    ):
        print(f"Training agent for {n_train_episodes} episodes...")
        self.run_name = f"{self.run_name}_{update_type}"

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

            if test_running_success_rate > 0.99999:
                if save_best:
                    if self.run_name is None:
                        print("WARNING: run_name is None, not saving best policy.")
                    else:
                        self.save_policy(self.run_name, save_best_dir)

                if early_stopping:
                    print(
                        f"CONVERGED: test success rate running avg reached 100% after {e} episodes."
                    )
                    break

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
