import multiprocessing
import time

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from DPAgent import DPAgent
from MCAgent import MCAgent

env_ver = "FrozenLake-v1"


def test_mc(i, seed):
    env = gym.make(
        env_ver, desc=generate_random_map(size=i, p=0.4, seed=seed), is_slippery=False
    )
    agent = MCAgent(env=env_ver, gamma=1.0, epsilon=0.4)
    agent.env = env
    agent.n_states, agent.n_actions = (
        agent.env.observation_space.n,
        agent.env.action_space.n,
    )
    agent.initialize()
    tic = time.perf_counter()
    trained = agent.train(
        max_steps=int((i**2) * 3),
        n_train_episodes=10_000,
        save_best=False,
        early_stopping=True,
        update_type="every_visit",
    )
    toc = time.perf_counter()
    return trained, toc - tic


def test_dp(i, seed, gamma=0.99):
    env = gym.make(env_ver, desc=generate_random_map(i, seed=seed), is_slippery=False)
    agent = DPAgent(env=env_ver, gamma=gamma)
    agent.env = env
    agent.V = np.zeros(agent.env.observation_space.n)
    agent.Pi = np.zeros(agent.env.observation_space.n, agent.env.action_space.n)
    agent.n_states, agent.n_actions = (
        agent.env.observation_space.n,
        agent.env.action_space.n,
    )

    return agent.train()


def run_test(i):
    mc_trained = False
    seed = 0
    mc_time = 0
    dp_time = 0
    while not mc_trained:
        seed = np.random.randint(0, 100000)
        mc_trained, train_time = test_mc(i, seed)
    mc_time = train_time
    dp_time = test_dp(i, seed)

    return mc_time, dp_time


def run_exp(gamma):
    times = []
    for i in range(8, 512, 8):
        # mc_time, dp_time = run_test(i)
        dp_time = test_dp(i, 0, gamma=gamma)
        times.append((i, dp_time))
    times = np.array(times)
    np.save(f"times_{gamma}.npy", times)
    return


def main():
    with multiprocessing.Pool(8) as p:
        gamma = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.1]
        p.map(run_exp, gamma)


if __name__ == "__main__":
    main()
