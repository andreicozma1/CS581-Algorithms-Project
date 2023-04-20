import os
import time
import numpy as np
import gradio as gr
from MonteCarloAgent import MonteCarloAgent


# For the dropdown list of policies
policies_folder = "policies"
all_policies = [file for file in os.listdir(policies_folder) if file.endswith(".npy")]

# All supported agents
agent_map = {
    "MonteCarloAgent": MonteCarloAgent,
    # TODO: Add DP Agent
}

# Global variables for the agent and the render fps (to allow changing it on the fly)
agent = None
render_fps_val = 5


def load_policy(policy_fname):
    print("Loading...")
    print(f"- policy_fname: {policy_fname}")
    global agent
    policy_path = os.path.join(policies_folder, policy_fname)
    props = policy_fname.split("_")
    agent_type, env_name = props[0], props[1]

    agent = agent_map[agent_type](env_name, render_mode="rgb_array")
    agent.load_policy(policy_path)

    return agent.env.spec.id, agent.__class__.__name__


def change_render_fps(x):
    print("Changing render fps:", x)
    global render_fps_val
    render_fps_val = x


def run(n_test_episodes, max_steps, render_fps):
    global agent, render_fps_val
    render_fps_val = render_fps
    print("Running...")
    print(f"- n_test_episodes: {n_test_episodes}")
    print(f"- max_steps: {max_steps}")
    print(f"- render_fps: {render_fps_val}")

    while agent is None:
        print("Waiting for agent to be loaded...")
        time.sleep(1)
        yield None, None, None, None, None, None, None, "🚫 ERROR: Please load a policy first!"

    rgb_array = np.random.random((25, 100, 3))
    episode, step = 0, 0
    state, action, reward = 0, 0, 0
    episodes_solved = 0

    def ep_str(episode):
        return f"{episode + 1} / {n_test_episodes} ({(episode + 1) / n_test_episodes * 100:.2f}%)"

    def step_str(step):
        return f"{step + 1}"

    for episode in range(n_test_episodes):
        for step, (episode_hist, solved, rgb_array) in enumerate(
            agent.generate_episode(max_steps=max_steps, render=True)
        ):
            if solved:
                episodes_solved += 1
            state, action, reward = episode_hist[-1]

            print(
                f"Episode: {ep_str(episode)} - step: {step_str} - state: {state} - action: {action} - reward: {reward} (frame time: {1 / render_fps:.2f}s)"
            )

            time.sleep(1 / render_fps_val)
            yield rgb_array, ep_str(episode), step_str(step), state, action, reward, ep_str(episodes_solved), "Running..."

    yield rgb_array, ep_str(episode), step_str(step), state, action, reward, ep_str(episodes_solved), "Done!"


with gr.Blocks() as demo:
    # TODO: Add title and description

    with gr.Row():
        with gr.Column():
            input_policy = gr.components.Dropdown(
                label="Policy", choices=all_policies, value=all_policies[0]
            )

            with gr.Row():
                out_environment = gr.components.Textbox(label="Environment")
                out_agent = gr.components.Textbox(label="Agent")

            btn_load = gr.components.Button("📁 Load")
            btn_load.click(
                fn=load_policy,
                inputs=[input_policy],
                outputs=[out_environment, out_agent],
            )

        with gr.Column():
            input_n_test_episodes = gr.components.Slider(
                minimum=1,
                maximum=100,
                value=5,
                label="Number of episodes",
            )
            input_max_steps = gr.components.Slider(
                minimum=1,
                maximum=500,
                value=500,
                label="Max steps per episode",
            )

            input_render_fps = gr.components.Slider(
                minimum=1,
                maximum=60,
                value=5,
                label="Render FPS",
            )
            input_render_fps.change(change_render_fps, inputs=[input_render_fps])

            btn_run = gr.components.Button("▶️ Run")

    out_msg = gr.components.Textbox(label="Message")

    with gr.Row():
        out_episode = gr.components.Textbox(label="Current Episode")
        out_step = gr.components.Textbox(label="Current Step")
        out_state = gr.components.Textbox(label="Current State")
        out_action = gr.components.Textbox(label="Chosen Action")
        out_reward = gr.components.Textbox(label="Reward Received")
        out_eps_solved = gr.components.Textbox(label="Episodes Solved")

    out_image = gr.components.Image(label="Environment", type="numpy", image_mode="RGB")

    btn_run.click(
        fn=run,
        inputs=[
            input_n_test_episodes,
            input_max_steps,
            input_render_fps,
        ],
        outputs=[
            out_image,
            out_episode,
            out_step,
            out_state,
            out_action,
            out_reward,
            out_eps_solved,
            out_msg,
        ],
    )


demo.queue(concurrency_count=2)
demo.launch()
