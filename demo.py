import os
import time
import numpy as np
import gradio as gr

import scipy.ndimage
import cv2

from agents import AGENTS_MAP

default_n_test_episodes = 10
default_max_steps = 500
default_render_fps = 5
default_epsilon = 0.0
default_paused = True

frame_env_h, frame_env_w = 512, 768
frame_policy_res = 256

# For the dropdown list of policies
policies_folder = "policies"
try:
    all_policies = [
        file for file in os.listdir(policies_folder) if file.endswith(".npy")
    ]
except FileNotFoundError:
    print("ERROR: No policies folder found!")
    all_policies = []


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


class RunState:
    def __init__(self) -> None:
        self.current_policy = None
        self.live_render_fps = default_render_fps
        self.live_epsilon = default_epsilon
        self.live_paused = default_paused
        self.live_steps_forward = None
        self.should_reset = False


def reset(state, policy_fname):
    if state.current_policy is not None and state.current_policy != policy_fname:
        state.should_reset = True
    state.live_paused = default_paused
    state.live_render_fps = default_render_fps
    state.live_epsilon = default_epsilon
    state.live_steps_forward = None
    return gr.update(value=pause_val_map_inv[not state.live_paused]), gr.update(
        interactive=state.live_paused
    )


def change_render_fps(state, x):
    print("Changing render fps:", x)
    state.live_render_fps = x
    return state


def change_epsilon(state, x):
    print("Changing greediness:", x)
    state.live_epsilon = x
    return state


def change_paused(state, x):
    print("Changing paused:", x)
    state.live_paused = pause_val_map[x]
    return (
        state,
        gr.update(value=pause_val_map_inv[not state.live_paused]),
        gr.update(interactive=state.live_paused),
    )


def onclick_btn_forward(state):
    print("Step forward")
    if state.live_steps_forward is None:
        state.live_steps_forward = 0
    state.live_steps_forward += 1
    return state


def run(
    localstate: RunState, policy_fname, n_test_episodes, max_steps, render_fps, epsilon
):
    localstate.current_policy = policy_fname
    localstate.live_render_fps = render_fps
    localstate.live_epsilon = epsilon
    localstate.live_steps_forward = None
    print("=" * 80)
    print("Running...")
    print(f"- policy_fname: {localstate.current_policy}")
    print(f"- n_test_episodes: {n_test_episodes}")
    print(f"- max_steps: {max_steps}")
    print(f"- render_fps: {localstate.live_render_fps}")
    print(f"- epsilon: {localstate.live_steps_forward}")

    policy_path = os.path.join(policies_folder, policy_fname)
    props = policy_fname.split("_")

    if len(props) < 2:
        yield localstate, None, None, None, None, None, None, None, None, None, None, "🚫 Please select a valid policy file."
        return

    agent_type, env_name = props[0], props[1]

    agent = AGENTS_MAP[agent_type](env=env_name, render_mode="rgb_array")
    agent.load_policy(policy_path)
    env_action_map = action_map.get(env_name)

    solved, frame_env, frame_policy = None, None, None
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
        time.sleep(0.25)

        for step, (episode_hist, solved, frame_env) in enumerate(
            agent.generate_episode(
                max_steps=max_steps,
                render=True,
                epsilon_override=localstate.live_epsilon,
            )
        ):
            _, _, last_reward = (
                episode_hist[-2] if len(episode_hist) > 1 else (None, None, None)
            )
            state, action, reward = episode_hist[-1]
            curr_policy = agent.Pi[state]

            frame_policy_h = frame_policy_res // len(curr_policy)
            frame_policy = np.zeros((frame_policy_h, frame_policy_res))
            for i, p in enumerate(curr_policy):
                frame_policy[
                    :,
                    i
                    * (frame_policy_res // len(curr_policy)) : (i + 1)
                    * (frame_policy_res // len(curr_policy)),
                ] = p

            frame_policy = scipy.ndimage.gaussian_filter(frame_policy, sigma=1.0)
            frame_policy = np.clip(
                frame_policy * (1.0 - localstate.live_epsilon)
                + localstate.live_epsilon / len(curr_policy),
                0.0,
                1.0,
            )

            label_loc_h, label_loc_w = frame_policy_h // 2, int(
                (action + 0.5) * frame_policy_res // len(curr_policy)
            )

            frame_policy_label_color = 1.0 - frame_policy[label_loc_h, label_loc_w]
            frame_policy_label_font = cv2.FONT_HERSHEY_SIMPLEX
            frame_policy_label_thicc = 1
            action_text_scale, action_text_label_scale = 0.6, 0.3

            (label_width, label_height), _ = cv2.getTextSize(
                str(action),
                frame_policy_label_font,
                action_text_scale,
                frame_policy_label_thicc,
            )

            cv2.putText(
                frame_policy,
                str(action),
                (
                    label_loc_w - label_width // 2,
                    label_loc_h + label_height // 2,
                ),
                frame_policy_label_font,
                action_text_scale,
                frame_policy_label_color,
                frame_policy_label_thicc,
                cv2.LINE_AA,
            )

            if env_action_map:
                action_name = env_action_map.get(action, "")
                (label_width, label_height), _ = cv2.getTextSize(
                    action_name,
                    frame_policy_label_font,
                    action_text_label_scale,
                    frame_policy_label_thicc,
                )

                cv2.putText(
                    frame_policy,
                    action_name,
                    (
                        int(label_loc_w - label_width / 2),
                        frame_policy_h
                        - (frame_policy_h - label_loc_h) // 2
                        + label_height // 2,
                    ),
                    frame_policy_label_font,
                    action_text_label_scale,
                    frame_policy_label_color,
                    frame_policy_label_thicc,
                    cv2.LINE_AA,
                )

            print(
                f"Episode: {ep_str(episode + 1)} - step: {step_str(step)} - state: {state} - action: {action} - reward: {reward} (epsilon: {localstate.live_epsilon:.2f}) (frame time: {1 / localstate.live_render_fps:.2f}s)"
            )

            yield localstate, agent_type, env_name, frame_env, frame_policy, ep_str(
                episode + 1
            ), ep_str(episodes_solved), step_str(
                step
            ), state, action, last_reward, "Running..."

            if localstate.live_steps_forward is not None:
                if localstate.live_steps_forward > 0:
                    localstate.live_steps_forward -= 1

                if localstate.live_steps_forward == 0:
                    localstate.live_steps_forward = None
                    localstate.live_paused = True
            else:
                time.sleep(1 / localstate.live_render_fps)

            while localstate.live_paused and localstate.live_steps_forward is None:
                yield localstate, agent_type, env_name, frame_env, frame_policy, ep_str(
                    episode + 1
                ), ep_str(episodes_solved), step_str(
                    step
                ), state, action, last_reward, "Paused..."
                time.sleep(1 / localstate.live_render_fps)
                if localstate.should_reset is True:
                    break

            if localstate.should_reset is True:
                localstate.should_reset = False
                yield (
                    localstate,
                    agent_type,
                    env_name,
                    np.ones((frame_env_h, frame_env_w, 3)),
                    np.ones((frame_policy_h, frame_policy_res)),
                    ep_str(episode + 1),
                    ep_str(episodes_solved),
                    step_str(step),
                    state,
                    action,
                    last_reward,
                    "Reset...",
                )
                return

        if solved:
            episodes_solved += 1

        time.sleep(0.25)

    localstate.current_policy = None
    yield localstate, agent_type, env_name, frame_env, frame_policy, ep_str(
        episode + 1
    ), ep_str(episodes_solved), step_str(step), state, action, reward, "Done!"


with gr.Blocks(title="CS581 Demo") as demo:
    gr.components.HTML(
        "<h1>CS581 Final Project Demo - Dynamic Programming & Monte-Carlo RL Methods (<a href='https://huggingface.co/spaces/acozma/CS581-Algos-Demo'>HF Space</a>)</h1>"
    )

    localstate = gr.State(RunState())

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

    btn_run = gr.components.Button("👀 Select & Load", interactive=bool(all_policies))

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
            label="Action Sampled vs Policy Distribution for Current State",
            type="numpy",
            image_mode="RGB",
        )
        out_image_policy.style(height=200)

    with gr.Row():
        input_epsilon = gr.components.Slider(
            minimum=0,
            maximum=1,
            value=default_epsilon,
            step=1 / 200,
            label="Epsilon (0 = greedy, 1 = random)",
        )
        input_epsilon.change(
            change_epsilon, inputs=[localstate, input_epsilon], outputs=[localstate]
        )

        input_render_fps = gr.components.Slider(
            minimum=1,
            maximum=60,
            value=default_render_fps,
            step=1,
            label="Simulation speed (fps)",
        )
        input_render_fps.change(
            change_render_fps,
            inputs=[localstate, input_render_fps],
            outputs=[localstate],
        )

    out_image_frame = gr.components.Image(
        label="Environment",
        type="numpy",
        image_mode="RGB",
    )
    out_image_frame.style(height=frame_env_h)

    with gr.Row():
        btn_pause = gr.components.Button(
            pause_val_map_inv[not default_paused], interactive=True
        )
        btn_forward = gr.components.Button("⏩ Step")

        btn_pause.click(
            fn=change_paused,
            inputs=[localstate, btn_pause],
            outputs=[localstate, btn_pause, btn_forward],
        )

        btn_forward.click(
            fn=onclick_btn_forward, inputs=[localstate], outputs=[localstate]
        )

    out_msg = gr.components.Textbox(
        value=""
        if all_policies
        else "ERROR: No policies found! Please train an agent first or add a policy to the policies folder.",
        label="Status Message",
    )

    input_policy.change(
        fn=reset,
        inputs=[localstate, input_policy],
        outputs=[localstate, btn_pause, btn_forward],
    )

    btn_run.click(
        fn=run,
        inputs=[
            localstate,
            input_policy,
            input_n_test_episodes,
            input_max_steps,
            input_render_fps,
            input_epsilon,
        ],
        outputs=[
            localstate,
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

demo.queue(concurrency_count=8)
demo.launch()
