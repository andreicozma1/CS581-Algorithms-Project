import os
import multiprocessing
import random

num_tests = 10
vals_eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
vals_gamma = [1.0, 0.99, 0.98, 0.97, 0.95]


def run_test(args):
    os.system(
        f"python3 MonteCarloAgent.py --train  --gamma {args[0]} --epsilon {args[1]} --wandb_project cs581 --wandb_job_type params --wandb_run_name_suffix {args[2]} --no_save"
    )


with multiprocessing.Pool(16) as p:
    tests = []
    for gamma in vals_gamma:
        for eps in vals_eps:
            tests.extend((gamma, eps, i) for i in range(num_tests))
    random.shuffle(tests)

    p.map(run_test, tests)
