"""Qwen-Image-Edit outpainting — extend room in all directions."""
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

print("Loading Qwen-Image-Edit...")
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16
)
pipeline.to("cuda")
print("Loaded!")

img = Image.open("/workspace/seva_input/room.png").convert("RGB")
print("Input:", img.size)

# Generate different views of the same room
prompts = [
    ("left_wall", "Show the left wall of this room. Same room, same style, same floor, same lighting. Show what is to the left of the current view."),
    ("right_wall", "Show the right wall of this room. Same room, same style, same floor, same lighting. Show what is to the right of the current view."),
    ("behind", "Show what is behind the camera in this room. Same room, same floor, same ceiling, same style. Turn around 180 degrees."),
    ("ceiling", "Show the ceiling of this room from below. Same room, same lighting."),
    ("wider_view", "Show a wider view of this living room, extend the image to show more of the room on both sides. Same style and lighting."),
]

for name, prompt in prompts:
    print("Generating:", name)
    inputs = {
        "image": [img],
        "prompt": prompt,
        "generator": torch.manual_seed(42),
        "true_cfg_scale": 4.0,
        "negative_prompt": "blurry, different room, different style, dark",
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
        result = output.images[0]
        path = "/workspace/qwen_room_%s.png" % name
        result.save(path)
        print("  Saved: %s (%dx%d)" % (path, result.size[0], result.size[1]))

print("Done! All views generated.")
