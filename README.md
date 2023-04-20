# CS581 Project - Reinforcement Learning: From Dynamic Programming to Monte-Carlo

[Google Slides](https://docs.google.com/presentation/d/1v4WwBQKoPnGiyCMXgUs-pCCJ8IwZqM3thUf-Ky00eTQ/edit?usp=sharing)

Evolution of Reinforcement Learning methods from pure Dynamic Programming-based methods to Monte Carlo methods + Bellman Optimization Comparison  

## Requirements

- Python 3
- Gymnasium: <https://pypi.org/project/gymnasium/>
- WandB: <https://pypi.org/project/wandb/>
- Gradio: <https://pypi.org/project/gradio/>

## Interactive Demo

TODO

## Dynamic-Programming Agent

TODO

### Usage

```bash
TODO
```

## Monte-Carlo Agent

The agent starts with a randomly initialized epsilon-greedy policy and uses either the first-visit or every-visit Monte-Carlo update method to learn the optimal policy.

Primarily tested on the [Cliff Walking](https://gymnasium.farama.org/environments/toy_text/cliff_walking/) toy environment.

```bash
# Training: Policy will be saved as a `.npy` file.
python3 MonteCarloAgent.py --train

# Testing: Use the `--test` flag with the path to the policy file.
python3 MonteCarloAgent.py --test policy_mc_CliffWalking-v0_e2000_s500_g0.99_e0.1.npy --render_mode human
```

### Usage

```bash
usage: MonteCarloAgent.py [-h] [--train] [--test TEST] [--n_train_episodes N_TRAIN_EPISODES] [--n_test_episodes N_TEST_EPISODES] [--test_every TEST_EVERY] [--max_steps MAX_STEPS] [--update_type {first_visit,every_visit}]
                          [--save_dir SAVE_DIR] [--no_save] [--gamma GAMMA] [--epsilon EPSILON] [--env ENV] [--render_mode RENDER_MODE] [--wandb_project WANDB_PROJECT] [--wandb_group WANDB_GROUP]
                          [--wandb_job_type WANDB_JOB_TYPE] [--wandb_run_name_suffix WANDB_RUN_NAME_SUFFIX]

options:
  -h, --help            show this help message and exit
  --train               Use this flag to train the agent.
  --test TEST           Use this flag to test the agent. Provide the path to the policy file.
  --n_train_episodes N_TRAIN_EPISODES
                        The number of episodes to train for. (default: 2000)
  --n_test_episodes N_TEST_EPISODES
                        The number of episodes to test for. (default: 100)
  --test_every TEST_EVERY
                        During training, test the agent every n episodes. (default: 100)
  --max_steps MAX_STEPS
                        The maximum number of steps per episode before the episode is forced to end. (default: 500)
  --update_type {first_visit,every_visit}
                        The type of update to use. (default: first_visit)
  --save_dir SAVE_DIR   The directory to save the policy to. (default: policies)
  --no_save             Use this flag to disable saving the policy.
  --gamma GAMMA         The value for the discount factor to use. (default: 0.99)
  --epsilon EPSILON     The value for the epsilon-greedy policy to use. (default: 0.1)
  --env ENV             The Gymnasium environment to use. (default: CliffWalking-v0)
  --render_mode RENDER_MODE
                        Render mode passed to the gym.make() function. Use 'human' to render the environment. (default: None)
  --wandb_project WANDB_PROJECT
                        WandB project name for logging. If not provided, no logging is done. (default: None)
  --wandb_group WANDB_GROUP
                        WandB group name for logging. (default: monte-carlo)
  --wandb_job_type WANDB_JOB_TYPE
                        WandB job type for logging. (default: train)
  --wandb_run_name_suffix WANDB_RUN_NAME_SUFFIX
                        WandB run name suffix for logging. (default: None)
```

## Presentation Guide

1. Title Slide: list the title of your talk along with your name  

2. Test Questions Slide: provide three questions relevant to your subject  

- short answers should suffice
- somewhere during your talk provide the answers, but do not emphasize them

3. Presenter’s Slides: let others get to know you  

- provide a little information about yourself, your degree program and your advisor  
- describe your interests and goals; show a map and picture(s) of your hometown
- as examples, students frequently like to mention their pets, their travels, their interests in music and food, even their favorite movies, you name it  

4. Outline Slide: provide a bulleted outline of the rest of your talk  

5. Overview Slide: list important definitions and provide a brief mention of applications  

6. History Slide: discuss major contributors, interesting stories and main developments  

7. Algorithms Slides: describe basic procedures and methodological comparisons  

- this should be the main part of your talk  
- discuss techniques from the most basic to the state-of-the-art  
- use examples and figures whenever possible  

8. Applications Slides: educate the class about amenable problems of interest to you  

- don’t get bogged down in too much minutiae  
- once again use examples and figures whenever possible  

9. Implementations Slides: discuss the results of your coding work (if any)  

- compare and contrast the algorithms you implemented
- make effective use of table and charts

10. Open Issues Slide: enumerate and discuss a few open questions

11. References Slide: provide a handful of key citations

12. Discussion Slide: solicit questions from the class

- this slide may have only a few bullets – it may even be left blank
- this is a good opportunity for other students to add to the discussion
- be ready to prompt some questions if there is silence
- remember not to repeat answers to your test questions

13. Test Questions Slide Revisited: show again your original test questions slide

- students may now complete their answer sheets and hand them to you
- Ashley will supervise as we applaud your excellent presentation!
