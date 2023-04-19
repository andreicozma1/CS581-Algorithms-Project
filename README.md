# CS581 Project - Reinforcement Learning: From Dynamic Programming to Monte-Carlo

[Google Slides](https://docs.google.com/presentation/d/1v4WwBQKoPnGiyCMXgUs-pCCJ8IwZqM3thUf-Ky00eTQ/edit?usp=sharing)

Evolution of Reinforcement Learning methods from pure Dynamic Programming-based methods to Monte Carlo methods + Bellman Optimization Comparison  

## Monte-Carlo Agent

The implementation of the epsilon-greedy Monte-Carlo agent for the [Cliff Walking](https://gymnasium.farama.org/environments/toy_text/cliff_walking/) toy environment.

### Training

```bash
python3 MonteCarloAgent.py --train
```

The final policy will be saved to a `.npy` file.

### Testing

Provide the path to the policy file as an argument to the `--test` flag.

```bash
python3 MonteCarloAgent.py --test policy_mc_CliffWalking-v0_e2000_s500_g0.99_e0.1.npy
```

### Visualization

```bash
python3 MonteCarloAgent.py --test policy_mc_CliffWalking-v0_e2000_s500_g0.99_e0.1.npy --render_mode human
```

### Default Parameters

```python
usage: MonteCarloAgent.py [-h] [--train] [--test TEST] [--n_train_episodes N_TRAIN_EPISODES] [--n_test_episodes N_TEST_EPISODES] [--test_every TEST_EVERY] [--max_steps MAX_STEPS] [--gamma GAMMA] [--epsilon EPSILON] [--env ENV]
                          [--render_mode RENDER_MODE] [--wandb_project WANDB_PROJECT] [--wandb_group WANDB_GROUP] [--wandb_job_type WANDB_JOB_TYPE]

options:
  -h, --help            show this help message and exit
  --train               Use this flag to train the agent. (default: False)
  --test TEST           Use this flag to test the agent. Provide the path to the policy file.
  --n_train_episodes N_TRAIN_EPISODES
                        The number of episodes to train for.
  --n_test_episodes N_TEST_EPISODES
                        The number of episodes to test for.
  --test_every TEST_EVERY
                        During training, test the agent every n episodes.
  --max_steps MAX_STEPS
                        The maximum number of steps per episode before the episode is forced to end.
  --gamma GAMMA         The value for the discount factor to use.
  --epsilon EPSILON     The value for the epsilon-greedy policy to use.
  --env ENV             The Gymnasium environment to use.
  --render_mode RENDER_MODE
                        The render mode to use. By default, no rendering is done. To render the environment, set this to 'human'.
  --wandb_project WANDB_PROJECT
                        WandB project name for logging. If not provided, no logging is done.
  --wandb_group WANDB_GROUP
                        WandB group name for logging. (default: monte-carlo)
  --wandb_job_type WANDB_JOB_TYPE
                        WandB job type for logging. (default: train)
```

## Presentation Guide (Text Version)

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
