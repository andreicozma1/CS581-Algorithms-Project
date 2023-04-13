import numpy as np
import gymnasium as gym
from tqdm import tqdm


def main():
    print("# Cliff Walking - Monte Carlo Train")
    env = gym.make("CliffWalking-v0")

    # Training parameters
    gamma, epsilon = 0.99, 0.1
    n_train_episodes, n_test_episodes, n_max_steps = 2000, 10, 500
    n_states, n_actions = env.observation_space.n, env.action_space.n
    print("=" * 80)
    print(f"gamma: {gamma}")
    print(f"epsilon: {epsilon}")
    print(f"n_episodes: {n_train_episodes}")
    print(f"n_steps: {n_max_steps}")
    print(f"n_states: {n_states}")
    print(f"n_actions: {n_actions}")
    print("=" * 80)

    # An arbitrary e-greedy policy
    Pi = np.full((n_states, n_actions), epsilon / n_actions)
    Pi[np.arange(n_states), np.random.randint(n_actions, size=n_states)] = (
        1 - epsilon + epsilon / n_actions
    )
    print("=" * 80)
    print("Initial policy:")
    print(Pi)
    print("=" * 80)
    Q = np.zeros((n_states, n_actions))
    R = [[[] for _ in range(n_actions)] for _ in range(n_states)]

    successes = []
    tqrange = tqdm(range(n_train_episodes))
    for i in tqrange:
        tqrange.set_description(f"Episode {i + 1:>4}")
        state, _ = env.reset()
        # Generate an episode following the current policy
        episode = []
        for _ in range(n_max_steps):
            # Randomly choose an action from the e-greedy policy
            action = np.random.choice(n_actions, p=Pi[state])
            # Take the action and observe the reward and next state
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            # This is where the agent got to the goal.
            # In the case in which agent jumped off the cliff, it is simply respawned at the start position without termination.
            if done:
                successes.append(1)
                break
        else:
            successes.append(0)

        G = 0
        # For each step of the episode, in reverse order
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            # Update the expected return
            G = gamma * G + reward
            # If we haven't already visited this state-action pair up to this point, then we can update the Q-table and policy
            # This is the first-visit MC method
            if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                R[state][action].append(G)
                Q[state, action] = np.mean(R[state][action])
                # e-greedy policy update
                Pi[state] = np.full(n_actions, epsilon / n_actions)
                # the greedy action is the one with the highest Q-value
                Pi[state, np.argmax(Q[state])] = 1 - epsilon + epsilon / n_actions

        success_rate_100 = np.mean(successes[-100:])
        success_rate_250 = np.mean(successes[-250:])
        success_rate_500 = np.mean(successes[-500:])
        tqrange.set_postfix(
            success_rate_100=f"{success_rate_100:.3f}",
            success_rate_250=f"{success_rate_250:.3f}",
            success_rate_500=f"{success_rate_500:.3f}",
        )

    print("Final policy:")
    print(Pi)
    np.save("policy.npy", Pi)

    print("=" * 80)
    print(f"Testing policy for {n_test_episodes} episodes...")
    # Test the policy for a few episodes
    env = gym.make("CliffWalking-v0", render_mode="human")
    for e in range(n_test_episodes):
        print(f"Test #{e + 1}:", end=" ")

        state, _ = env.reset()
        for _ in range(n_max_steps):
            action = np.random.choice(n_actions, p=Pi[state])
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            if done:
                print("Success!")
                break
        else:
            print("Failed!")

    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
