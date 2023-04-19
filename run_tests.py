import os
import multiprocessing

vals_eps = [0.1, 0.25, 0.5, 0.75, 0.9]
vals_gamma = [1.0, 0.97, 0.95, 0.9, 0.75, 0.5]

num_tests = 10


def run_test(args):
    os.system(
        f"python3 MonteCarloAgent.py --train  --gamma {args[0]} --epsilon {args[1]} --wandb_project cs581 --wandb_job_type params --wandb_run_name_suffix {args[2]}"
    )


with multiprocessing.Pool(8) as p:
    # make all the tests
    tests = []
    for gamma in vals_gamma:
        for eps in vals_eps:
            tests.extend((gamma, eps, i) for i in range(num_tests))
    p.map(run_test, tests)
