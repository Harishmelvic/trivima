"""
Full 360 panorama from single room photo via Qwen-Image-Edit iterative outpainting.
With color correction to prevent brightness drift.
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
ow, oh = original.size
print("Original: %dx%d" % (ow, oh))

# Reference stats from original for color correction
orig_stats = ImageStat.Stat(original)
orig_mean = orig_stats.mean
orig_std = [s**0.5 for s in orig_stats.var]


def color_correct(img, ref_mean, ref_std):
    """Match color distribution to reference image."""
    arr = np.array(img).astype(np.float32)
    for c in range(3):
        ch = arr[:, :, c]
        ch_mean = ch.mean()
        ch_std = ch.std() + 1e-6
        # Normalize then match to reference distribution
        arr[:, :, c] = (ch - ch_mean) / ch_std * ref_std[c] + ref_mean[c]
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def outpaint(img, direction, extend_ratio, prompt):
    """Outpaint image in given direction."""
    w, h = img.size
    extend_px = int(w * extend_ratio) if direction in ("left", "right") else int(h * extend_ratio)

    if direction == "left":
        canvas = Image.new("RGB", (w + extend_px, h), (200, 200, 200))
        canvas.paste(img, (extend_px, 0))
    elif direction == "right":
        canvas = Image.new("RGB", (w + extend_px, h), (200, 200, 200))
        canvas.paste(img, (0, 0))
    elif direction == "up":
        canvas = Image.new("RGB", (w, h + extend_px), (200, 200, 200))
        canvas.paste(img, (0, extend_px))
    elif direction == "down":
        canvas = Image.new("RGB", (w, h + extend_px), (200, 200, 200))
        canvas.paste(img, (0, 0))

    inputs = {
        "image": [canvas],
        "prompt": prompt,
        "generator": torch.manual_seed(42),
        "true_cfg_scale": 4.0,
        "negative_prompt": "blurry, dark, overexposed, different room, different style, low quality, artifacts",
        "num_inference_steps": 40,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
        result = output.images[0]

    # Color correct to match original
    result = color_correct(result, orig_mean, orig_std)
    return result


# Build panorama by extending left and right alternately
current = original.copy()
# Resize to consistent height
target_h = 512
current = current.resize((int(ow * target_h / oh), target_h), Image.LANCZOS)

steps = [
    ("left",  0.4, "Continue this room to the left showing more wall, doorway, and room corner. Same room style and floor."),
    ("right", 0.4, "Continue this room to the right showing more wall, curtains, and room corner. Same room style and floor."),
    ("left",  0.3, "Continue extending the room left. Show the adjacent wall or hallway entrance. Same floor and ceiling."),
    ("right", 0.3, "Continue extending the room right. Show the room corner and adjacent wall. Same floor and ceiling."),
    ("left",  0.3, "Continue the room further left. Show the back wall starting to appear. Same room, same style."),
    ("right", 0.3, "Continue the room further right. Show the back wall from the other side. Same room, same style."),
    ("left",  0.25, "Complete the room view to the left. Show the back wall behind where the camera was. Same room."),
    ("right", 0.25, "Complete the room view to the right. Connect back to the starting view. Same room."),
]

for i, (direction, ratio, prompt) in enumerate(steps):
    print("[%d/%d] Extending %s (%.0f%%)..." % (i+1, len(steps), direction, ratio*100))
    current = outpaint(current, direction, ratio, prompt)
    path = "/workspace/pano360_step%d.png" % (i+1)
    current.save(path)
    print("  Result: %dx%d -> %s" % (current.size[0], current.size[1], path))

# Save final panorama
current.save("/workspace/pano360_final.png")
print("\nFinal 360 panorama: %dx%d" % current.size)
print("Saved: /workspace/pano360_final.png")
