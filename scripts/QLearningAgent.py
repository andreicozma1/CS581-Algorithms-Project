import numpy as np


class QLearningAgent:
    def __init__(self, epsilon, alpha, gamma, num_state, num_actions, action_space):
        """
        Constructor
        Args:
            epsilon: The degree of exploration
            gamma: The discount factor
            num_state: The number of states
            num_actions: The number of actions
            action_space: To call the random action
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_state = num_state
        self.num_actions = num_actions

        self.Q = np.zeros((self.num_state, self.num_actions))
        self.action_space = action_space

    def update(self, state, state2, reward, action, action2):
        """
        Update the action value function using the Q-Learning update.
        Q(S, A) = Q(S, A) + alpha(reward + (gamma * Q(S_, A_) - Q(S, A))
        Args:
            prev_state: The previous state
            next_state: The next state
            reward: The reward for taking the respective action
            prev_action: The previous action
            next_action: The next action
        Returns:
            None
        """
        predict = self.Q[state, action]
        target = reward + self.gamma * np.max(self.Q[state2, :])
        self.Q[state, action] += self.alpha * (target - predict)


episode = [
    ["s1", "a1", -8],
    ["s1", "a2", -16],
    ["s2", "a1", 20],
    ["s1", "a2", -10],
    ["s2", "a1", None],
]

index_map = {
    "s1": 0,
    "s2": 1,
    "a1": 0,
    "a2": 1,
}


def main_r():
    print("# QLearningAgent.py")
    agent = QLearningAgent(0.1, 0.5, 0.5, 2, 2, [0, 1])
    print(agent.Q)
    for i in range(len(episode) - 1):
        print(f"# Step {i + 1}")
        s, a, r = episode[i]
        s2, a2, r2 = episode[i + 1]
        agent.update(index_map[s], index_map[s2], r, index_map[a], index_map[a2])
        print(agent.Q)


if __name__ == "__main__":
    main_r()
