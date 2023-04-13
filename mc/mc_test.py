import numpy as np
import gymnasium as gym
from tqdm import tqdm

policy_file = "policy.npy"
n_steps = 500
n_test_episodes = 10


def main():
    print("=" * 80)
    print("# Cliff Walking - Monte Carlo Test")
    print("=" * 80)
    # save the policy
    print(f"Loading policy from file: '{policy_file}'...")
    Pi = np.load(policy_file)
    print("Policy:")
    print(Pi)
    print(f"shape: {Pi.shape}")
    _, n_actions = Pi.shape

    print("=" * 80)
    print(f"Testing policy for {n_test_episodes} episodes...")
    env = gym.make("CliffWalking-v0", render_mode="human")
    for e in range(n_test_episodes):
        print(f"Test #{e + 1}:", end=" ")

        state, _ = env.reset()
        for _ in range(n_steps):
            action = np.random.choice(n_actions, p=Pi[state])
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            if done:
                print("Success!")
                break
        else:
            print("Failed!")

    env.close()


if __name__ == "__main__":
    main()
