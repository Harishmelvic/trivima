"""
Qwen-Image-Edit iterative outpainting → 360° panorama.

Strategy: place original image in center of a wider canvas,
mask the empty areas, let Qwen fill them with room context.
Repeat: shift and extend in each direction.
Finally stitch into a panorama strip.
"""
import torch
import numpy as np
from PIL import Image
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

def outpaint_direction(img, direction, extend_px, prompt_hint):
    """Extend image in a direction by placing it offset on a larger canvas."""
    w, h = img.size

    if direction == "left":
        new_w = w + extend_px
        canvas = Image.new("RGB", (new_w, h), (128, 128, 128))
        canvas.paste(img, (extend_px, 0))
        prompt = "Continue this living room to the left. Same wall color, same floor, same style, same lighting. %s" % prompt_hint
    elif direction == "right":
        new_w = w + extend_px
        canvas = Image.new("RGB", (new_w, h), (128, 128, 128))
        canvas.paste(img, (0, 0))
        prompt = "Continue this living room to the right. Same wall color, same floor, same style, same lighting. %s" % prompt_hint
    elif direction == "up":
        new_h = h + extend_px
        canvas = Image.new("RGB", (w, new_h), (128, 128, 128))
        canvas.paste(img, (0, extend_px))
        prompt = "Continue this room upward showing the ceiling. Same room, same lighting. %s" % prompt_hint
    elif direction == "down":
        new_h = h + extend_px
        canvas = Image.new("RGB", (w, new_h), (128, 128, 128))
        canvas.paste(img, (0, 0))
        prompt = "Continue this room downward showing more floor. Same room, same floor material. %s" % prompt_hint

    print("  Canvas: %dx%d, prompt: %s" % (canvas.size[0], canvas.size[1], prompt[:60]))

    inputs = {
        "image": [canvas],
        "prompt": prompt,
        "generator": torch.manual_seed(42),
        "true_cfg_scale": 4.0,
        "negative_prompt": "blurry, dark, different room, different style, low quality",
        "num_inference_steps": 40,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
        result = output.images[0]

    return result

# Step 1: Extend left
print("\n[1/4] Extending LEFT...")
extended_left = outpaint_direction(original, "left", ow // 2, "Show the left wall and corner.")
extended_left.save("/workspace/pano_step1_left.png")
print("  Result: %dx%d" % extended_left.size)

# Step 2: Extend right from the left-extended image
print("\n[2/4] Extending RIGHT...")
extended_lr = outpaint_direction(extended_left, "right", ow // 2, "Show the right wall and window area.")
extended_lr.save("/workspace/pano_step2_lr.png")
print("  Result: %dx%d" % extended_lr.size)

# Step 3: Extend left again (further)
print("\n[3/4] Extending LEFT further...")
extended_wider = outpaint_direction(extended_lr, "left", ow // 2, "Show more of the room continuing left. Wall and furniture.")
extended_wider.save("/workspace/pano_step3_wider.png")
print("  Result: %dx%d" % extended_wider.size)

# Step 4: Extend right again
print("\n[4/4] Extending RIGHT further...")
panorama = outpaint_direction(extended_wider, "right", ow // 2, "Show more of the room continuing right. Complete the panoramic view.")
panorama.save("/workspace/pano_final.png")
print("  Final panorama: %dx%d" % panorama.size)

# Also save the original for comparison
original.save("/workspace/pano_original.png")

print("\nDone! Panorama saved to /workspace/pano_final.png")
print("Steps saved: pano_step1_left.png, pano_step2_lr.png, pano_step3_wider.png")
