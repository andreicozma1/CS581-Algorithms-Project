import numpy as np
from tqdm import tqdm
import wandb
from AgentBase import AgentBase


class MCAgent(AgentBase):
    def __init__(
        self, /, update_type="on-policy", **kwargs  # "on-policy" or "off-policy
    ):
        super().__init__(run_name=self.__class__.__name__, **kwargs)
        self.update_type = update_type
        self.run_name = f"{self.run_name}_{self.update_type}"
        self.initialize()

    def initialize(self):
        print("Resetting all state variables...")
        # The Q-Table holds the current expected return for each state-action pair
        # random uniform initialization
        self.Q = np.random.uniform(-1, 1, size=(self.n_states, self.n_actions))
        # other alternatives:
        # self.Q = np.zeros((self.n_states, self.n_actions))
        # self.Q = np.random.rand(self.n_states, self.n_actions)
        # self.Q = np.random.normal(0, 1, size=(self.n_states, self.n_actions))

        if self.update_type.startswith("on_policy"):
            # For On-Policy update type:
            # R keeps track of all the returns that have been observed for each state-action pair to update Q
            self.R = [[[] for _ in range(self.n_actions)] for _ in range(self.n_states)]
            # An arbitrary e-greedy policy:
            self.Pi = self.create_soft_policy()
        elif self.update_type.startswith("off_policy"):
            # For Off-Policy update type:
            self.C = np.zeros((self.n_states, self.n_actions))
            # Target policy is greedy with respect to the current Q (ties broken consistently)
            self.Pi = np.zeros((self.n_states, self.n_actions))
            self.Pi[np.arange(self.n_states), np.argmax(self.Q, axis=1)] = 1.0
            # Behavior policy is e-greedy with respect to the current Q
            self.Pi_behaviour = self.create_soft_policy(coverage_policy=self.Pi)
        else:
            raise ValueError(
                f"update_type must be either 'on_policy' or 'off_policy', but got {self.update_type}"
            )
        print("=" * 80)
        print("Initial policy:")
        print(self.Pi)
        print("=" * 80)

    def create_soft_policy(self, coverage_policy=None):
        """
        Create a soft policy (epsilon-greedy).
        If coverage_policy is None, the soft policy is initialized randomly.
        Otherwise, the soft policy is e-greedy with respect to the coverage policy. (useful for off-policy)
        """
        # With probability epsilon, sample an action uniformly at random
        Pi = np.full((self.n_states, self.n_actions), self.epsilon / self.n_actions)
        # The greedy action receives the remaining probability mass
        # If coverage_policy is not provided, the greedy action is sampled randomly
        # Otherwise we give the remaining probability mass according to the coverage policy
        Pi[
            np.arange(self.n_states),
            np.random.randint(self.n_actions, size=self.n_states)
            if coverage_policy is None
            else np.argmax(coverage_policy, axis=1),
        ] = (
            1.0 - self.epsilon + self.epsilon / self.n_actions
        )
        return Pi

    def update_on_policy(self, episode_hist):
        G = 0.0
        # For each step of the episode, in reverse order
        for t in range(len(episode_hist) - 1, -1, -1):
            state, action, reward = episode_hist[t]
            # Updating the expected return
            G = self.gamma * G + reward
            # First-visit MC method:
            # Updating the expected return and policy only if this is the first visit to this state-action pair
            if (state, action) not in [(x[0], x[1]) for x in episode_hist[:t]]:
                self.R[state][action].append(G)
                self.Q[state, action] = np.mean(self.R[state][action])
                # Updating the epsilon-greedy policy.
                # With probability epsilon, sample an action uniformly at random
                self.Pi[state] = np.full(self.n_actions, self.epsilon / self.n_actions)
                # The greedy action receives the remaining probability mass
                self.Pi[state, np.argmax(self.Q[state])] = (
                    1 - self.epsilon + self.epsilon / self.n_actions
                )

    # def update_every_visit(self, episode_hist):
    #     G = 0
    #     # Backward pass through the trajectory
    #     for t in range(len(episode_hist) - 1, -1, -1):
    #         state, action, reward = episode_hist[t]
    #         # Updating the expected return
    #         G = self.gamma * G + reward
    #         # Every-visit MC method:
    #         # Updating the expected return and policy for every visit to this state-action pair
    #         self.R[state][action].append(G)
    #         self.Q[state, action] = np.mean(self.R[state][action])
    #         # Updating the epsilon-greedy policy.
    #         # With probability epsilon, sample an action uniformly at random
    #         self.Pi[state] = np.full(self.n_actions, self.epsilon / self.n_actions)
    #         # The greedy action receives the remaining probability mass
    #         self.Pi[state, np.argmax(self.Q[state])] = (
    #             1 - self.epsilon + self.epsilon / self.n_actions
    #         )

    def update_off_policy(self, episode_hist):
        G, W = 0.0, 1.0
        for t in range(len(episode_hist) - 1, -1, -1):
            state, action, reward = episode_hist[t]
            # Updating the expected return
            G = self.gamma * G + reward
            self.C[state, action] = self.C[state, action] + W
            self.Q[state, action] = self.Q[state, action] + (
                W / self.C[state, action]
            ) * (G - self.Q[state, action])
            # Updating the target policy to be greedy with respect to the current Q
            greedy_action = np.argmax(self.Q[state])
            self.Pi[state] = np.zeros(self.n_actions)
            self.Pi[state, greedy_action] = 1.0
            # If the greedy action is not the action taken by the behavior policy, then break
            if action != greedy_action:
                break
            W = W * (1.0 / self.Pi_behaviour[state, action])

        # Update the behavior policy such that it has coverage of the target policy
        self.Pi_behaviour = self.create_soft_policy(coverage_policy=self.Pi)

    def train(
        self,
        n_train_episodes=2000,
        test_every=100,
        log_wandb=False,
        save_best=True,
        save_best_dir=None,
        early_stopping=False,
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

        update_func = getattr(self, f"update_{self.update_type}")

        tqrange = tqdm(range(n_train_episodes))
        tqrange.set_description("Training")

        if log_wandb:
            self.wandb_log_img(episode=None)

        for e in tqrange:
            policy = self.Pi_behaviour if self.update_type == "off_policy" else self.Pi
            episode_hist, solved, _ = self.run_episode(policy=policy, **kwargs)
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

            # Test the agent every test_every episodes
            if test_every > 0 and e % test_every == 0:
                # For off policy, self.Pi is the target policy. For on policy, self.Pi is the soft policy
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

            if test_running_success_rate > 0.99:
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
