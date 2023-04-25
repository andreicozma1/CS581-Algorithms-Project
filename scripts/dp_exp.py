import multiprocessing
import time

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

import wandb
from DPAgent import DPAgent
from MCAgent import MCAgent

env_ver = "FrozenLake-v1"


def test_dp(gamma=0.99):
    env = gym.make(
        env_ver,
        render_mode="ansi",
        # desc=generate_random_map(8, seed=3141),
        # is_slippery=False,
    )
    dp = DPAgent(env=env_ver, gamma=0.99)
    dp.env = env
    dp.env_name = env_ver
    dp.V = np.zeros(dp.env.observation_space.n)
    dp.Pi = np.zeros(dp.env.observation_space.n, dp.env.action_space.n)
    dp.n_states, dp.n_actions = (
        dp.env.observation_space.n,
        dp.env.action_space.n,
    )
    times = dp.train()

    # np.save(f"times_{gamma}.npy", times)
    s = env.render()
    print(s)


def main():
    wandb.init(
        project="cs581",
        # job_type=args.wandb_job_type,
        # config=dict(args._get_kwargs()),
    )
    np.set_printoptions(linewidth=500, precision=3)
    # with multiprocessing.Pool(8) as p:
    #     gamma = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.1]
    #     p.map(test_dp, gamma)
    test_dp(0.99)


if __name__ == "__main__":
    main()
