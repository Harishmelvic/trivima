"""Generate multi-view images from single photo using Zero123++."""
import torch
from PIL import Image

print("Loading Zero123++ via custom pipeline...")
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
print("Loaded!")

img = Image.open("/workspace/trivima/test_room.jpg").convert("RGB")
print("Input:", img.size)

print("Generating 6 views...")
result = pipe(img, num_inference_steps=30)
output = result.images[0]
print("Output:", output.size)

# Save the 6-view composite
output.save("/workspace/trivima/output_gaussian/zero123_composite.png")

# Split into individual views (Zero123++ outputs a 3x2 grid)
w, h = output.size
vw, vh = w // 3, h // 2
for row in range(2):
    for col in range(3):
        view = output.crop((col * vw, row * vh, (col+1) * vw, (row+1) * vh))
        idx = row * 3 + col
        view.save("/workspace/trivima/output_gaussian/zero123_view_%d.png" % idx)
        print("  View %d: %dx%d" % (idx, view.size[0], view.size[1]))

print("Done! 6 synthetic views generated from 1 photo")

del pipe
torch.cuda.empty_cache()
