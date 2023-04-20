import os
import multiprocessing
import random

num_tests = 5

update_types = ["first_visit", "every_visit"]
vals_eps = [0.1, 0.25, 0.5, 0.75, 0.9]
vals_gamma = [1.0, 0.99, 0.98, 0.97, 0.96, 0.95]


def run_test(args):
    os.system(
        f"python3 MonteCarloAgent.py --train  --gamma {args[0]} --epsilon {args[1]} --update_type {args[2]} --wandb_project cs581 --wandb_job_type params --wandb_run_name_suffix {args[3]} --no_save"
    )


with multiprocessing.Pool(8) as p:
    tests = []
    for update_type in update_types:
        for gamma in vals_gamma:
            for eps in vals_eps:
                tests.extend((gamma, eps, update_type, i) for i in range(num_tests))
    random.shuffle(tests)

    p.map(run_test, tests)
