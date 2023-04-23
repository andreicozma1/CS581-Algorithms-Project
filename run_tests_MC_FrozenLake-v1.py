import os
import multiprocessing
import random

wandb_project = "cs581"

env = "FrozenLake-v1"
n_train_episodes = 5000
max_steps = 200

num_tests = 10

vals_update_type = ["first_visit"]
vals_epsilon = [0.1, 0.2, 0.3, 0.4, 0.5]
vals_gamma = [1.0, 0.98, 0.96, 0.94]


def run_test(args):
    os.system(
        f"python3 run.py --agent MCAgent --train --n_train_episodes {n_train_episodes} --max_steps {max_steps} --env {env} --gamma {args[0]} --epsilon {args[1]} --update_type {args[2]} --wandb_project {wandb_project} --wandb_run_name_suffix {args[3]} --no_save"
    )


with multiprocessing.Pool(16) as p:
    tests = []
    for update_type in vals_update_type:
        for gamma in vals_gamma:
            for eps in vals_epsilon:
                tests.extend((gamma, eps, update_type, i) for i in range(num_tests))
    random.shuffle(tests)

    p.map(run_test, tests)
