"""
Fixed 360 panorama: maintain resolution by resizing after each step.
Outpaint left and right, keep height at 512px.
"""
import torch
import numpy as np
from PIL import Image, ImageStat
from diffusers import QwenImageEditPlusPipeline

print("Loading Qwen-Image-Edit...")
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")
print("Loaded!")

original = Image.open("/workspace/seva_input/room.png").convert("RGB")
TARGET_H = 512

# Resize original to target height
ow, oh = original.size
scale = TARGET_H / oh
original = original.resize((int(ow * scale), TARGET_H), Image.LANCZOS)
ow, oh = original.size
print("Original resized: %dx%d" % (ow, oh))

# Color stats for correction
orig_arr = np.array(original).astype(np.float32)
orig_mean = [orig_arr[:,:,c].mean() for c in range(3)]
orig_std = [orig_arr[:,:,c].std() for c in range(3)]


def color_correct(img):
    arr = np.array(img).astype(np.float32)
    for c in range(3):
        ch = arr[:,:,c]
        ch_mean = ch.mean()
        ch_std = ch.std() + 1e-6
        arr[:,:,c] = (ch - ch_mean) / ch_std * orig_std[c] + orig_mean[c]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def outpaint_and_fix(img, direction, extend_px, prompt):
    w, h = img.size

    if direction == "left":
        canvas = Image.new("RGB", (w + extend_px, h), (180, 180, 180))
        canvas.paste(img, (extend_px, 0))
    else:
        canvas = Image.new("RGB", (w + extend_px, h), (180, 180, 180))
        canvas.paste(img, (0, 0))

    inputs = {
        "image": [canvas],
        "prompt": prompt,
        "generator": torch.manual_seed(42),
        "true_cfg_scale": 4.0,
        "negative_prompt": "blurry, dark, overexposed, different room, low quality",
        "num_inference_steps": 40,
    }
    with torch.inference_mode():
        result = pipeline(**inputs).images[0]

    # Force resize back to target height, maintain aspect ratio
    rw, rh = result.size
    new_w = int(rw * TARGET_H / rh)
    result = result.resize((new_w, TARGET_H), Image.LANCZOS)

    # Color correct
    result = color_correct(result)
    return result


current = original.copy()
extend_px = int(ow * 0.35)  # extend 35% each step

steps = [
    ("left",  "Continue this room to the left. Same beige walls, same hardwood floor, same modern style. Show left wall corner."),
    ("right", "Continue this room to the right. Same walls, same floor, same curtains. Show right side of room."),
    ("left",  "Extend room further left. Show the left wall with door or hallway. Same room style and lighting."),
    ("right", "Extend room further right. Show right wall corner. Same floor texture and wall color."),
    ("left",  "Continue room left. Show the wall behind where camera was. Same style, completing the room."),
    ("right", "Continue room right. Show the opposite wall. Same room, same lighting, same floor."),
]

for i, (direction, prompt) in enumerate(steps):
    print("[%d/%d] Extending %s..." % (i+1, len(steps), direction))
    current = outpaint_and_fix(current, direction, extend_px, prompt)
    path = "/workspace/pano_fixed_step%d.png" % (i+1)
    current.save(path)
    print("  %dx%d saved" % current.size)

current.save("/workspace/pano_fixed_final.png")
print("\nFinal panorama: %dx%d" % current.size)
print("Saved: /workspace/pano_fixed_final.png")
