import os
import wandb

from PIL import Image
from PIL.Image import Transpose, Resampling


api = wandb.Api()
run = api.run("acozma/cs581/5ttfkav8")

print(run.summary)
print("Downloading images...")

for file in run.files():
    if file.name.endswith(".png"):
        file.download(exist_ok=True)

print("Finished downloading images")


def process_images(image_fnames, upscale=20):
    image_fnames.sort(key=lambda x: int(x.split("_")[-2]))
    frames = [Image.open(image) for image in image_fnames]
    frames = [frame.transpose(Transpose.ROTATE_90) for frame in frames]
    frames = [
        frame.resize(
            (frame.size[0] * upscale, frame.size[1] * upscale),
            resample=Resampling.NEAREST,
        )
        for frame in frames
    ]
    return frames


def images_to_gif(frames, fname, duration=500):
    print(f"Creating gif: {fname}")

    frame_one = frames[0]
    frame_one.save(
        f"{fname}.gif",
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=duration,
        loop=0,
    )


folder_path = "./media/images"
all_fnames = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]


fnames_policy = [f for f in all_fnames if os.path.basename(f).startswith("Policy")]
policy_frames = process_images(fnames_policy)

fnames_qtable = [f for f in all_fnames if os.path.basename(f).startswith("Q-table")]
qtable_frames = process_images(fnames_qtable)

spacing_factor = 1 / 2

final_frames = []
for i, (qtable, policy) in enumerate(zip(qtable_frames, policy_frames)):
    width, height = qtable.size
    final_height = int(height * 2 + height * spacing_factor)
    new_frame = Image.new("RGB", (width, final_height), color="white")
    new_frame.paste(qtable, (0, 0))
    new_frame.paste(policy, (0, height + int(height * spacing_factor)))
    final_frames.append(new_frame)

images_to_gif(final_frames, "qtable_policy")
