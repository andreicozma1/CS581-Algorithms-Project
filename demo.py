import os
import time
from matplotlib import interactive
import numpy as np
import gradio as gr
from MonteCarloAgent import MonteCarloAgent
import scipy.ndimage

# For the dropdown list of policies
policies_folder = "policies"
try:
    all_policies = [
        file for file in os.listdir(policies_folder) if file.endswith(".npy")
    ]
except FileNotFoundError:
    print("ERROR: No policies folder found!")
    all_policies = []

# All supported agents
agent_map = {
    "MonteCarloAgent": MonteCarloAgent,
    # TODO: Add DP Agent
}

# Global variables to allow changing it on the fly
live_render_fps = 10
live_epsilon = 0.0
live_paused = False


def change_render_fps(x):
    print("Changing render fps:", x)
    global live_render_fps
    live_render_fps = x


def change_epsilon(x):
    print("Changing greediness:", x)
    global live_epsilon
    live_epsilon = x


def change_paused(x):
    print("Changing paused:", x)
    global live_paused
    live_paused = x
    # change the text to resume
    return gr.update(value="▶️ Resume" if x else "⏸️ Pause")


def run(policy_fname, n_test_episodes, max_steps, render_fps, epsilon):
    global live_render_fps, live_epsilon
    live_render_fps = render_fps
    live_epsilon = epsilon
    print("Running...")
    print(f"- n_test_episodes: {n_test_episodes}")
    print(f"- max_steps: {max_steps}")
    print(f"- render_fps: {live_render_fps}")

    policy_path = os.path.join(policies_folder, policy_fname)
    props = policy_fname.split("_")
    agent_type, env_name = props[0], props[1]

    agent = agent_map[agent_type](env_name, render_mode="rgb_array")
    agent.load_policy(policy_path)

    rgb_array = None
    policy_viz = None
    episode, step = 0, 0
    state, action, reward = 0, 0, 0
    episodes_solved = 0

    def ep_str(episode):
        return f"{episode + 1} / {n_test_episodes} ({(episode + 1) / n_test_episodes * 100:.2f}%)"

    def step_str(step):
        return f"{step + 1}"

    for episode in range(n_test_episodes):
        for step, (episode_hist, solved, rgb_array) in enumerate(
            agent.generate_episode(
                max_steps=max_steps, render=True, override_epsilon=True
            )
        ):
            while live_paused:
                time.sleep(0.1)

            if solved:
                episodes_solved += 1
            state, action, reward = episode_hist[-1]
            curr_policy = agent.Pi[state]

            viz_w, viz_h = 128, 16
            policy_viz = np.zeros((viz_h, viz_w))
            for i, p in enumerate(curr_policy):
                policy_viz[
                    :,
                    i
                    * (viz_w // len(curr_policy)) : (i + 1)
                    * (viz_w // len(curr_policy)),
                ] = p

            policy_viz = scipy.ndimage.gaussian_filter(policy_viz, sigma=1)
            policy_viz = np.clip(
                policy_viz * (1 - live_epsilon) + live_epsilon / len(curr_policy), 0, 1
            )

            print(
                f"Episode: {ep_str(episode)} - step: {step_str(step)} - state: {state} - action: {action} - reward: {reward} (frame time: {1 / render_fps:.2f}s)"
            )

            time.sleep(1 / live_render_fps)
            # Live-update the agent's epsilon value for demonstration purposes
            agent.epsilon = live_epsilon
            yield agent_type, env_name, rgb_array, policy_viz, ep_str(episode), ep_str(
                episodes_solved
            ), step_str(step), state, action, reward, "Running..."

    yield agent_type, env_name, rgb_array, policy_viz, ep_str(episode), ep_str(
        episodes_solved
    ), step_str(step), state, action, reward, "Done!"


with gr.Blocks(title="CS581 Demo") as demo:
    gr.components.HTML(
        "<h1>Reinforcement Learning: From Dynamic Programming to Monte-Carlo (Demo)</h1>"
    )
    gr.components.HTML("<h3>Authors: Andrei Cozma and Landon Harris</h3>")

    gr.components.HTML("<h2>Select Configuration:</h2>")
    with gr.Row():
        input_policy = gr.components.Dropdown(
            label="Policy Checkpoint",
            choices=all_policies,
            value=all_policies[0] if all_policies else "No policies found :(",
        )

        out_environment = gr.components.Textbox(label="Resolved Environment")
        out_agent = gr.components.Textbox(label="Resolved Agent")

    with gr.Row():
        input_n_test_episodes = gr.components.Slider(
            minimum=1,
            maximum=500,
            value=500,
            label="Number of episodes",
        )
        input_max_steps = gr.components.Slider(
            minimum=1,
            maximum=500,
            value=500,
            label="Max steps per episode",
        )

    btn_run = gr.components.Button(
        "▶️ Start", interactive=True if all_policies else False
    )

    gr.components.HTML("<h2>Live Statistics & Policy Visualization:</h2>")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                out_episode = gr.components.Textbox(label="Current Episode")
                out_step = gr.components.Textbox(label="Current Step")
                out_eps_solved = gr.components.Textbox(label="Episodes Solved")

            with gr.Row():
                out_state = gr.components.Textbox(label="Current State")
                out_action = gr.components.Textbox(label="Chosen Action")
                out_reward = gr.components.Textbox(label="Reward Received")

        out_image_policy = gr.components.Image(
            value=np.ones((16, 128)),
            label="policy[state]",
            type="numpy",
            image_mode="RGB",
        )

    gr.components.HTML("<h2>Live Customization:</h2>")
    with gr.Row():
        input_epsilon = gr.components.Slider(
            minimum=0,
            maximum=1,
            value=live_epsilon,
            label="Epsilon (0 = greedy, 1 = random)",
        )
        input_epsilon.change(change_epsilon, inputs=[input_epsilon])

        input_render_fps = gr.components.Slider(
            minimum=1, maximum=60, value=live_render_fps, label="Simulation speed (fps)"
        )
        input_render_fps.change(change_render_fps, inputs=[input_render_fps])

    out_image_frame = gr.components.Image(
        label="Environment", type="numpy", image_mode="RGB"
    )

    with gr.Row():
        # Pause/resume button
        btn_pause = gr.components.Button("⏸️ Pause", interactive=True)
        btn_pause.click(
            fn=change_paused,
            inputs=[btn_pause],
            outputs=[btn_pause],
        )

    out_msg = gr.components.Textbox(
        value=""
        if all_policies
        else "<h2>🚫 ERROR: No policies found! Please train an agent first or add a policy to the policies folder.<h2>",
        label="Status Message",
    )

    btn_run.click(
        fn=run,
        inputs=[
            input_policy,
            input_n_test_episodes,
            input_max_steps,
            input_render_fps,
            input_epsilon,
        ],
        outputs=[
            out_agent,
            out_environment,
            out_image_frame,
            out_image_policy,
            out_episode,
            out_eps_solved,
            out_step,
            out_state,
            out_action,
            out_reward,
            out_msg,
        ],
    )

demo.queue(concurrency_count=3)
demo.launch()
