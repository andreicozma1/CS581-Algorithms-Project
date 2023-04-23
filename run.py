import argparse
import wandb

from utils import AGENTS_MAP, load_agent


def main():
    parser = argparse.ArgumentParser()
    ### Train/Test parameters
    parser.add_argument(
        "--train",
        action="store_true",
        help="Use this flag to train the agent.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Use this flag to test the agent. Provide the path to the policy file.",
    )
    parser.add_argument(
        "--n_train_episodes",
        type=int,
        default=2500,
        help="The number of episodes to train for. (default: 2500)",
    )
    parser.add_argument(
        "--n_test_episodes",
        type=int,
        default=100,
        help="The number of episodes to test for. (default: 100)",
    )
    parser.add_argument(
        "--test_every",
        type=int,
        default=100,
        help="During training, test the agent every n episodes. (default: 100)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="The maximum number of steps per episode before the episode is forced to end. (default: 200)",
    )

    ### Agent parameters
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=AGENTS_MAP.keys(),
        help=f"The agent to use. Currently supports one of: {list(AGENTS_MAP.keys())}",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="The value for the discount factor to use. (default: 0.99)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.4,
        help="The value for the epsilon-greedy policy to use. (default: 0.4)",
    )

    parser.add_argument(
        "--update_type",
        type=str,
        choices=["first_visit", "every_visit"],
        default="first_visit",
        help="The type of update to use. Only supported by Monte-Carlo agent. (default: first_visit)",
    )

    ### Environment parameters
    parser.add_argument(
        "--env",
        type=str,
        default="CliffWalking-v0",
        choices=["CliffWalking-v0", "FrozenLake-v1", "Taxi-v3"],
        help="The Gymnasium environment to use. (default: CliffWalking-v0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The seed to use when generating the FrozenLake environment. If not provided, a random seed is used. (default: None)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=8,
        help="The size to use when generating the FrozenLake environment. (default: 8)",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default=None,
        help="Render mode passed to the gym.make() function. Use 'human' to render the environment. (default: None)",
    )

    # Logging and saving parameters
    parser.add_argument(
        "--save_dir",
        type=str,
        default="policies",
        help="The directory to save the policy to. (default: policies)",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Use this flag to disable saving the policy.",
    )
    parser.add_argument(
        "--run_name_suffix",
        type=str,
        default=None,
        help="Run name suffix for logging and policy checkpointing. (default: None)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="WandB project name for logging. If not provided, no logging is done. (default: None)",
    )
    parser.add_argument(
        "--wandb_job_type",
        type=str,
        default="train",
        help="WandB job type for logging. (default: train)",
    )

    args = parser.parse_args()
    print(vars(args))

    agent = load_agent(
        args.agent if args.test is None else args.test, **dict(args._get_kwargs())
    )

    agent.run_name += f"_e{args.n_train_episodes}_s{args.max_steps}"
    if args.run_name_suffix is not None:
        agent.run_name += f"+{args.run_name_suffix}"

    try:
        if args.train:
            # Log to WandB
            if args.wandb_project is not None:
                wandb.init(
                    project=args.wandb_project,
                    name=agent.run_name,
                    group=args.agent,
                    job_type=args.wandb_job_type,
                    config=dict(args._get_kwargs()),
                )

            agent.train(
                n_train_episodes=args.n_train_episodes,
                test_every=args.test_every,
                n_test_episodes=args.n_test_episodes,
                max_steps=args.max_steps,
                update_type=args.update_type,
                log_wandb=args.wandb_project is not None,
                save_best=True,
                save_best_dir=args.save_dir,
            )
            if not args.no_save:
                agent.save_policy(save_dir=args.save_dir)
        elif args.test is not None:
            agent.test(
                n_test_episodes=args.n_test_episodes,
                max_steps=args.max_steps,
            )
        else:
            print("ERROR: Please provide either --train or --test.")
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()
