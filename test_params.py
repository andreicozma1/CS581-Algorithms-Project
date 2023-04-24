import argparse
import os
import multiprocessing
import random


# argument parsing
parser = argparse.ArgumentParser(description="Run parameter tests for MC agent")
parser.add_argument(
    "--env",
    type=str,
    default="FrozenLake-v1",
    help="environment to run",
)
parser.add_argument(
    "--num_tests",
    type=int,
    default=10,
    help="number of tests to run for each parameter combination",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default=None,
    help="wandb project name to log to",
)

args = parser.parse_args()

env, num_tests, wandb_project = args.env, args.num_tests, args.wandb_project
agent = "MCAgent"

vals_update_type = [
    # "on_policy",
    "off_policy",
]  # Note: Every visit takes too long due to these environment's reward structure
# vals_gamma = [1.0, 0.98, 0.96, 0.94]
vals_epsilon = [0.1, 0.2, 0.3, 0.4, 0.5]
vals_gamma = [1.0]
# vals_epsilon = [0.5]

vals_size = [8, 16, 32, 64]

if env == "CliffWalking-v0":
    n_train_episodes = 2500
    # max_steps = 200
elif env == "FrozenLake-v1":
    n_train_episodes = 25000
    # max_steps = 200
elif env == "Taxi-v3":
    n_train_episodes = 10000
    # max_steps = 500
else:
    raise ValueError(f"Unsupported environment: {env}")


def run_test(args):
    command = f"python3 run.py --train --agent {agent} --env {env}"
    # command += f" --n_train_episodes {n_train_episodes} --max_steps {max_steps}"
    command += f" --n_train_episodes {n_train_episodes}"
    for k, v in args.items():
        command += f" --{k} {v}"
    if wandb_project is not None:
        command += f" --wandb_project {wandb_project}"
    command += " --no_save"
    os.system(command)


with multiprocessing.Pool(8) as p:
    tests = []
    for update_type in vals_update_type:
        for gamma in vals_gamma:
            for eps in vals_epsilon:
                if env == "FrozenLake-v1":
                    for size in vals_size:
                        tests.extend(
                            {
                                "gamma": gamma,
                                "epsilon": eps,
                                "update_type": update_type,
                                "size": size,
                                "run_name_suffix": i,
                            }
                            for i in range(num_tests)
                        )
                else:
                    tests.extend(
                        {
                            "gamma": gamma,
                            "epsilon": eps,
                            "update_type": update_type,
                            "run_name_suffix": i,
                        }
                        for i in range(num_tests)
                    )
    random.shuffle(tests)

    p.map(run_test, tests)
