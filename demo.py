import os
import time
import numpy as np
import gradio as gr
from MonteCarloAgent import MonteCarloAgent
import scipy.ndimage
import cv2

default_n_test_episodes = 10
default_max_steps = 500

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
action_map = {
    "CliffWalking-v0": {
        0: "up",
        1: "right",
        2: "down",
        3: "left",
    },
    "FrozenLake-v1": {
        0: "left",
        1: "down",
        2: "right",
        3: "up",
    },
}


pause_val_map = {
    "▶️ Resume": False,
    "⏸️ Pause": True,
}
pause_val_map_inv = {v: k for k, v in pause_val_map.items()}

# Global variables to allow changing it on the fly
live_render_fps = 5
live_epsilon = 0.0
live_paused = True
live_steps_forward = None
should_reset = False


# def reset():
#     global should_reset
#     should_reset = True


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
    live_paused = pause_val_map[x]
    next_val = pause_val_map_inv[not live_paused]
    return gr.update(value=next_val), gr.update(interactive=live_paused)


def onclick_btn_forward():
    print("Step forward")
    global live_steps_forward
    if live_steps_forward is None:
        live_steps_forward = 0
    live_steps_forward += 1


def run(policy_fname, n_test_episodes, max_steps, render_fps, epsilon):
    global live_render_fps, live_epsilon, live_paused, live_steps_forward, should_reset
    live_render_fps = render_fps
    live_epsilon = epsilon
    live_steps_forward = None
    print("=" * 80)
    print("Running...")
    print(f"- policy_fname: {policy_fname}")
    print(f"- n_test_episodes: {n_test_episodes}")
    print(f"- max_steps: {max_steps}")
    print(f"- render_fps: {live_render_fps}")
    print(f"- epsilon: {live_epsilon}")

    policy_path = os.path.join(policies_folder, policy_fname)
    props = policy_fname.split("_")

    if len(props) < 2:
        yield None, None, None, None, None, None, None, None, None, None, "🚫 Please select a valid policy file."
        return

    agent_type, env_name = props[0], props[1]

    agent = agent_map[agent_type](env_name, render_mode="rgb_array")
    agent.load_policy(policy_path)
    env_action_map = action_map.get(env_name)

    solved, rgb_array, policy_viz = None, None, None
    episode, step, state, action, reward, last_reward = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    episodes_solved = 0

    def ep_str(episode):
        return (
            f"{episode} / {n_test_episodes} ({(episode) / n_test_episodes * 100:.2f}%)"
        )

    def step_str(step):
        return f"{step + 1}"

    for episode in range(n_test_episodes):
        time.sleep(0.5)

        for step, (episode_hist, solved, rgb_array) in enumerate(
            agent.generate_episode(
                max_steps=max_steps, render=True, epsilon_override=live_epsilon
            )
        ):
            _, _, last_reward = (
                episode_hist[-2] if len(episode_hist) > 1 else (None, None, None)
            )
            state, action, reward = episode_hist[-1]
            curr_policy = agent.Pi[state]

            rgb_array_height, rgb_array_width = 384, 768
            rgb_array = cv2.resize(
                rgb_array,
                (
                    int(rgb_array.shape[1] / rgb_array.shape[0] * rgb_array_height),
                    rgb_array_height,
                ),
                interpolation=cv2.INTER_AREA,
            )

            if rgb_array.shape[1] < rgb_array_width:
                rgb_array_new = np.pad(
                    rgb_array,
                    (
                        (0, 0),
                        (
                            (rgb_array_width - rgb_array.shape[1]) // 2,
                            (rgb_array_width - rgb_array.shape[1]) // 2,
                        ),
                        (0, 0),
                    ),
                    "constant",
                )
                rgb_array = np.uint8(rgb_array_new)

            viz_w = 512
            viz_h = viz_w // len(curr_policy)
            policy_viz = np.zeros((viz_h, viz_w))
            for i, p in enumerate(curr_policy):
                policy_viz[
                    :,
                    i
                    * (viz_w // len(curr_policy)) : (i + 1)
                    * (viz_w // len(curr_policy)),
                ] = p

            policy_viz = scipy.ndimage.gaussian_filter(policy_viz, sigma=1.0)
            policy_viz = np.clip(
                policy_viz * (1.0 - live_epsilon) + live_epsilon / len(curr_policy),
                0.0,
                1.0,
            )

            cv2.putText(
                policy_viz,
                str(action),
                (
                    int((action + 0.5) * viz_w // len(curr_policy) - 8),
                    viz_h // 2 - 10,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                1.0,
                2,
                cv2.LINE_AA,
            )

            if env_action_map:
                action_name = env_action_map.get(action, "")

                cv2.putText(
                    policy_viz,
                    action_name,
                    (
                        int(
                            (action + 0.5) * viz_w // len(curr_policy)
                            - 5 * len(action_name)
                        ),
                        viz_h // 2 + 20,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    1.0,
                    2,
                    cv2.LINE_AA,
                )

            print(
                f"Episode: {ep_str(episode + 1)} - step: {step_str(step)} - state: {state} - action: {action} - reward: {reward} (epsilon: {live_epsilon:.2f}) (frame time: {1 / live_render_fps:.2f}s)"
            )

            yield agent_type, env_name, rgb_array, policy_viz, ep_str(
                episode + 1
            ), ep_str(episodes_solved), step_str(
                step
            ), state, action, last_reward, "Running..."

            if live_steps_forward is not None:
                if live_steps_forward > 0:
                    live_steps_forward -= 1

                if live_steps_forward == 0:
                    live_steps_forward = None
                    live_paused = True
            else:
                time.sleep(1 / live_render_fps)

            while live_paused and live_steps_forward is None:
                yield agent_type, env_name, rgb_array, policy_viz, ep_str(
                    episode + 1
                ), ep_str(episodes_solved), step_str(
                    step
                ), state, action, last_reward, "Paused..."
                time.sleep(1 / live_render_fps)
            #     if should_reset is True:
            #         break

            # if should_reset is True:
            #     should_reset = False
            #     return (
            #         agent_type,
            #         env_name,
            #         rgb_array,
            #         policy_viz,
            #         ep_str(episode + 1),
            #         ep_str(episodes_solved),
            #         step_str(step),
            #         state,
            #         action,
            #         last_reward,
            #         "Resetting...",
            #     )

        if solved:
            episodes_solved += 1

        time.sleep(0.5)

    yield agent_type, env_name, rgb_array, policy_viz, ep_str(episode + 1), ep_str(
        episodes_solved
    ), step_str(step), state, action, reward, "Done!"


with gr.Blocks(title="CS581 Demo") as demo:
    gr.components.HTML(
        "<h1>CS581 Final Project Demo - Reinforcement Learning: From Dynamic Programming to Monte-Carlo</h1>"
    )

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
            maximum=1000,
            value=default_n_test_episodes,
            label="Number of episodes",
        )
        input_max_steps = gr.components.Slider(
            minimum=1,
            maximum=1000,
            value=default_max_steps,
            label="Max steps per episode",
        )

    btn_run = gr.components.Button("👀 Select", interactive=bool(all_policies))

    gr.components.HTML("<h2>Live Visualization & Information:</h2>")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                out_episode = gr.components.Textbox(label="Current Episode")
                out_step = gr.components.Textbox(label="Current Step")
                out_eps_solved = gr.components.Textbox(label="Episodes Solved")

            with gr.Row():
                out_state = gr.components.Textbox(label="Current State")
                out_action = gr.components.Textbox(label="Chosen Action")
                out_reward = gr.components.Textbox(label="Last Reward")

        out_image_policy = gr.components.Image(
            value=np.ones((16, 128)),
            label="Action Sampled vs Policy Distribution for Current State",
            type="numpy",
            image_mode="RGB",
        )

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
        btn_pause = gr.components.Button(
            pause_val_map_inv[not live_paused], interactive=True
        )
        btn_forward = gr.components.Button("⏩ Step")

        btn_pause.click(
            fn=change_paused,
            inputs=[btn_pause],
            outputs=[btn_pause, btn_forward],
        )

        btn_forward.click(
            fn=onclick_btn_forward,
        )

    out_msg = gr.components.Textbox(
        value=""
        if all_policies
        else "ERROR: No policies found! Please train an agent first or add a policy to the policies folder.",
        label="Status Message",
    )

    # input_policy.change(fn=reset)

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
