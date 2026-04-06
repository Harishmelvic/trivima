from diffusers import DiffusionPipeline
import inspect, torch

pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

sig = inspect.signature(pipe.__call__)
print("Pipeline class:", type(pipe).__name__)
print("Parameters:")
for name, param in sig.parameters.items():
    default = str(param.default) if param.default != param.empty else "REQUIRED"
    print("  %s: %s" % (name, default[:50]))
